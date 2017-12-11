#!/usr/bin/python2

from __future__ import absolute_import, print_function

import argparse
import io
import logging
import os
import sys
import time
from logging import debug, info

import cPickle
import numpy as np
import re
import scipy

import tornado.ioloop
import tornado.websocket
import tornado.httpserver
import tornado.template
import tornado.web
import webrtcvad
from tornado.web import url
import json

import nexmo
from os.path import join, dirname
from dotenv import load_dotenv


#Only used for record function
import datetime
import wave
import scipy.io.wavfile as wav

CLIP_MIN_MS = 2000  # 200ms - the minimum audio clip that will be used
MAX_LENGTH = 10000  # Max length of a sound clip for processing in ms
SILENCE = 20  # How many continuous frames of silence determine the end of a phrase
path = './recordings/'

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
MODEL_PATH = os.getenv('MODEL_PATH')

HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
event_url = "http://{}/event".format(HOST)


# Constants:
BYTES_PER_FRAME = 640  # Bytes in a frame
MS_PER_FRAME = 20  # Duration of a frame in ms

CLIP_MIN_FRAMES = CLIP_MIN_MS // MS_PER_FRAME

# Global variables
conns = {}
uuid = None
Predict = None

class BufferedPipe(object):
    def __init__(self, max_frames, sink):
        """
        Create a buffer which will call the provided `sink` when full.

        It will call `sink` with the number of frames and the accumulated bytes when it reaches
        `max_buffer_size` frames.
        """
        self.sink = sink
        self.max_frames = max_frames

        self.count = 0
        self.payload = b''

    def append(self, data, cli):
        """ Add another data to the buffer. `data` should be a `bytes` object. """

        self.count += 1
        self.payload += data

        if self.count == self.max_frames:
            self.process(cli)

    def process(self, cli):
        """ Process and clear the buffer. """

        self.sink(self.count, self.payload, cli)
        self.count = 0
        self.payload = b''


class Processor(object):
    def __init__(self, path, is_predicting):
        self.path = path
        self.is_predicting = is_predicting
    def process(self, count, payload, cli):
        print("count {}".format(count))
        print("self.is_predicting {}".format(self.is_predicting))
        if count > CLIP_MIN_FRAMES:  # If the buffer is less than CLIP_MIN_MS, ignore it
            if self.is_predicting == False:
                info('Processing {} frames from {}'.format(count, cli))
                fn = "{}rec-{}-{}.wav".format(self.path, cli, datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
                output = wave.open(fn, 'wb')
                output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                output.writeframes(payload)
                output.close()
                info('File written {}'.format(fn))

                prediction, prob = Predict.perdict(fn)
                print("prediction {}".format(prediction))
                print("uuid {}".format(uuid))
                print("self.is_predicting {}".format(self.is_predicting))
                count = 0
                payload = None
                fs, audio = wav.read("prediction/{}.wav".format(prediction))
                self.playback(audio.tobytes(), cli)
            else:
                print("playback")
                self.playback(payload, cli)
                self.is_predicting = False
        else:
            info('Discarding {} frames'.format(str(count)))
    def playback(self, content, cli):
        self.is_predicting = False
        frames = len(content) // 640
        info("Playing {} frames to {}".format(frames, cli))
        conn = conns[cli]
        pos = 0
        for x in range(0, frames + 1):
            newpos = pos + 640
            debug("writing bytes {} to {} to socket for {}".format(pos, newpos, cli))
            data = content[pos:newpos]
            conn.write_message(data, binary=True)
            time.sleep(0.018)
            pos = newpos


class Predict(object):
    def __init__(self, modelPath):
        self.files = ["pills_1","pills_10", "pills_25", "pills_50"]
        self.model = self.loadSVModel(modelPath)

    def loadSVModel(self,SVMmodePath):
        '''
        This function loads an SVM model either for classification or training.
        ARGMUMENTS:
            - SVMmodelName:     the path of the model to be loaded
            - isRegression:        a flag indigating whereas this model is regression or not
        '''
        print("loading model {}".format(SVMmodePath))

        try:
            fo = open(SVMmodePath+"MEANS", "rb")
        except IOError:
                print ("Load SVM Model: Didn't find file")
                return
        try:
            MEAN = cPickle.load(fo)
            STD = cPickle.load(fo)

        except:
            fo.close()
        fo.close()

        MEAN = np.array(MEAN)
        STD = np.array(STD)

        COEFF = []
        with open(SVMmodePath, 'rb') as fid:
            SVM = cPickle.load(fid)

        return(SVM, MEAN, STD)

    def perdict(self, file):
        shortTermWindow = 0.050
        shortTermStep = 0.050
        from _pyAudioAnalysis import audioBasicIO
        from _pyAudioAnalysis import audioFeatureExtraction as aF
        print("perdict: files:{} file:{} ".format(self.files, file ))
        #read audio file, convert to mono (if needed)
        [Fs, x] = audioBasicIO.readAudioFile(file)
        x = audioBasicIO.stereo2mono(x)

        mtWin, mtStep, stWin, stStep = 1.0, 1.0, shortTermWindow, shortTermStep
        Classifier, MEAN, STD = self.model
       

        #extract features from sample
        [MidTermFeatures, s] = aF.mtFeatureExtraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs * stWin), round(Fs * stStep))
        MidTermFeatures = MidTermFeatures.mean(axis=1)        # long term averaging of mid-term statistics
        curFV = (MidTermFeatures - MEAN) / STD                # normalization

        #predict
        result = Classifier.predict(curFV.reshape(1,-1))[0]
        prob = Classifier.predict_proba(curFV.reshape(1,-1))[0]

        s = self.files[int(result)]

        return s, prob

class WSHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, processor):
        # Create a buffer which will call `process` when it is full:
        self.frame_buffer = BufferedPipe(MAX_LENGTH // MS_PER_FRAME, processor)
        # Setup the Voice Activity Detector
        self.tick = None
        self.cli = None
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)  # Level of sensitivity
    def open(self):
        info("client connected")
        # Add the connection to the list of connections
        self.tick = 0
    def on_message(self, message):
        # Check if message is Binary or Text
        if type(message) == str:
            if self.vad.is_speech(message, 16000):
                debug ("SPEECH from {}".format(self.cli))
                self.tick = SILENCE
                self.frame_buffer.append(message, self.cli)
            else:
                debug("Silence from {} TICK: {}".format(self.cli, self.tick))
                self.tick -= 1
                if self.tick == 0:
                    self.frame_buffer.process(self.cli)  # Force processing and clearing of the buffer
        else:
            info(message)
            # Here we should be extracting the meta data that was sent and attaching it to the connection object
            data = json.loads(message)
            self.cli = data['cli']
            conns[self.cli] = self
            self.write_message('ok')

    def on_close(self):
        # Remove the connection from the list of connections
        del conns[self.cli]
        info("client disconnected")


class NCCOHandler(tornado.web.RequestHandler):
    def initialize(self, host, event_url):
        self._host = host
        self._event_url = event_url
        self._template = tornado.template.Loader(".").load("ncco.json")
        info("template {}".format(self._template))
    def get(self):
        cli = self.get_argument("from", None).lstrip("+")
        to = self.get_argument("to", None)
        conv_uuid = self.get_argument("conversation_uuid", None)
        self.set_header("Content-Type", 'application/json')
        self.write(self._template.generate(
            host=self._host,
            event_url=self._event_url,
            lvn = to,
            cli = cli
        ))
        self.finish()


class EventHandler(tornado.web.RequestHandler):
    def post(self):
        info(self.request.body)
        data = json.loads(self.request.body)
        global uuid
        uuid = data['uuid']

        # print("got uuid {}".format(uuid))
        # print("data {}".format(data))
        self.set_header("Content-Type", 'text/plain')
        self.write('ok')
        self.finish()


def main(argv=sys.argv[1:]):
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--verbose", action="count")
        ap.add_argument("-c", "--config", default=None)

        args = ap.parse_args(argv)

        logging.basicConfig(
            level=logging.INFO if args.verbose < 1 else logging.DEBUG,
            format="%(levelname)7s %(message)s",
        )

        global Predict 
        Predict = Predict(MODEL_PATH)

        is_predicting = False


        #Pass any config for the processor into this argument.
        processor = Processor(path, is_predicting).process

        application = tornado.web.Application([
            url(r"/ncco", NCCOHandler, dict(host=HOST, event_url=event_url)),
            url(r'/socket', WSHandler, dict(processor=processor)),
            url(r'/event', EventHandler),
        ])

        http_server = tornado.httpserver.HTTPServer(application)
        http_server.listen(PORT)
        info("Running on port %s", PORT)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        pass  # Suppress the stack-trace on quit


if __name__ == "__main__":
    main()
