import asyncio
from time import monotonic
import time
import websockets
import json
import threading
from six.moves import queue
import webrtcvad
import requests
import collections
import base64
import requests
import math
import wave
# from requests.packages.urllib3.exceptions import InsecureRequestWarning
 
# requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
from rediscluster import RedisCluster
startup_nodes1=[
        {
            "port": "6380",
            "host": "10.150.0.72",
        },
        {
            "port": "6379",
            "host": "10.150.0.72",
        }, {
            "port": "6382",
            "host": "10.150.0.72",
        },
        {
            "port": "6381",
            "host": "10.150.0.72",
        },
        {
            "port": "6384",
            "host": "10.150.0.72",
        },
        {
            "port": "6383",
            "host": "10.150.0.72",
        }
    ]
rc = RedisCluster(startup_nodes=startup_nodes1, decode_responses=True)

class Transcoder(object):
    """
    Converts audio chunks to text
    """
    RATE = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50
    RATE_PROCESS = 16000
    def __init__(self,meeting_id,connection_id,deviceid,input_rate=RATE_PROCESS):
        self.buff = queue.Queue()
        self.input_rate = input_rate
        self.input_rate = 16000
        self.vad = webrtcvad.Vad(3)
        self.deviceid1 =deviceid
        self.time=time.time()
        self.meeting_id = meeting_id
        self.connection_id= connection_id
        self.block_size = int(self.RATE_PROCESS /
                              float(self.BLOCKS_PER_SECOND))
    frame_duration_ms = property(
    lambda self: 1000 * self.block_size // self.input_rate)
    
    
    def start1(self):
        """Start up streaming speech call"""
        t = threading.Thread(target=self.process)
        # t.daemon = True
        t.start()

    async def get_text(self,audio,fromat):
        print("api its called")   
        data_re=rc.get(self.meeting_id)
        kl=json.loads(data_re)
        device_id=kl["devicesInfo"]
        device_id1=device_id[self.deviceid1]
        speaker_lag = device_id1["speaker_lang"]
        # print(speaker_lag)
        data = {}
        encoded = base64.b64encode(audio) 
        data['voice'] = encoded.decode('utf-8')
        data['extra_key'] =speaker_lag 
        data['meeting_id']=self.meeting_id
        data['connection_id']=self.connection_id
        data['device_id']=self.deviceid1
        data["speakerName"]=device_id1["name"]
        data["isHost"]=device_id1["isHost"]
        data["Cc_State"]=device_id1["closedCaption"]
        data["isFinal"]=fromat
        data["devicesInfo"]=kl["devicesInfo"]              
        # # print(data["devicesInfo"],type(data["devicesInfo"]) )   
        result = requests.post("http://192.168.13.51:8000/transcribe2",json=data) 
        return result.json()

    def write_wav(self, filename, data):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        # assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.input_rate)
        wf.writeframes(data)
        wf.close()

    def process(self):
        print("Listening (ctrl-C to exit)...")
        frames = self.stream_generator()
        wav_data = bytearray()
        try:
            for frame in frames:
                if frame is not None:
                    wav_data.extend(frame)
                else:
                    start = time.perf_counter()
                    # self.write_wav("harvard6.wav",wav_data)
                    text =asyncio.run(self.get_text(wav_data,True))
                    if text is not None:
                        inference_time = time.perf_counter()-start
                        print(f"{inference_time:.3f}s\t{text}")
                    wav_data = bytearray()
        except KeyboardInterrupt:
            exit()

    def stream_generator(self, padding_ms=300, ratio=0.75, frames=None,sample_rate =16000):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|"""
 
        frames=None
        if frames is None:
            frames =  self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
 
        for frame in frames:
            if len(frame) < 640:
                return
 
            is_speech = self.vad.is_speech(frame, sample_rate)
 
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()
 
            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
 
                    ring_buffer.clear()

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == 16000:
            while True:
                yield self.write()
        else:
            raise Exception("Resampling required")
   
   
    def write(self):
       
        return self.buff.get()
 
import threading

clients = set()

async def handle_client(websocket):
    """
    Collects audio from the stream, writes it to buffer and return the output of  speech to text
    """
    print('New client. Websocket ID = %s. We now have %d clients' % (id(websocket), len(clients)))

    config = await websocket.recv()
    config = json.loads(config)
    # print(config)
    transcoder = Transcoder(meeting_id=config["meeting_id"],connection_id=config["connection_id"],
                            deviceid=config["device_id"])
    transcoder.start1()

    try:
        async for message in websocket:
            await asyncio.get_running_loop().run_in_executor(None, transcoder.buff.put, message)
            await websocket.send("received")
    except websockets.exceptions.ConnectionClosedOK as error:
        print(f"<- Client connection finalized.", error)
    except websockets.ConnectionClosedError as error:
        print('Websocket: Client connection failed.', error)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        clients.remove(websocket)
        print("Client Remove")

async def audio_processor(websocket):
    clients.add(websocket)
    await handle_client(websocket)

async def start_server():
    async with websockets.serve(audio_processor, "192.168.13.51", 5544, ping_timeout=25000,ping_interval=20000):
        print("WebSocket server started")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    async def main():
        try:
            await start_server()
        except KeyboardInterrupt:
            pass
        finally:
            print("WebSocket server stopped")

    asyncio.run(main())


