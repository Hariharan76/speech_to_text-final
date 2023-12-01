from fastapi import FastAPI,Request,Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import base64
import io
import re
import speech_recognition as sr
from pydub import AudioSegment
# from better_profanity import profanity
import warnings
import aiohttp
warnings.filterwarnings("ignore")
from faster_whisper import WhisperModel
import torch
import warnings
from rediscluster import RedisCluster
import asyncio 
warnings.filterwarnings("ignore")
import time
import json
import uvicorn
import logging

# logging.basicConfig()
# logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
# model_path = "whisper-large-v2-ct2/"
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
async def trans(data, das):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://192.168.13.51:8008/translation", json=data) as response:
            result = await response.json()
            das.update(result) 
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]
 
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
class transCribe(BaseModel):
 
    voice:bytes
    extra_key:str
    meeting_id:str
    connection_id:str
    device_id:str
    speakerName:str
    isHost:bool
    Cc_State:bool
    devicesInfo:dict
    isFinal:bool
 
@app.get("/")
def read_root():
    return {"Voice2Text"}
 
 
blacklist = [
    "",
    "Thank you!",
    "Thank you",
    "Thanks for",
    "Thank you.",
    "  Thank you.",
    "you",
     "Bye.",
    "Bye!",
    "Bye bye!",
    "lease sub",
    "The end.",
    "I love you.",
    "",
    "Thanks for watching!",
    "Thank you for watching!",
    "Thanks for watching.",
    "Thank you for watching.",
    "Please subscribe!"
    "Please subscribe to my channel!",
    "Please subscribe to my channel.",
    "Please search TongTongTV online.",
    "Please search TongTongTV online!",
    "Please search TongTongTV online",
    "you",
    "Go to Beadaholique.com for all of your beading supply needs!",
    "MBC 뉴스 이재경입니다."
    "This is the end of this video. Thank you for watching."
    "Thanks for watching and don't forget to like and subscribe!"
]
# make all list entries lowercase for later comparison
blacklist = list((map(lambda x: x.lower(), blacklist)))
model_path = "/home/vectone/Documents/Audio_speker/whisper-large-v2-ct2"
if torch.cuda.is_available():
# Run on GPU with FP16
    model = WhisperModel(model_path, device="cuda", compute_type="float16", num_workers=1)
    print("GPU Device")
else:
    print("CPU Deivice")


def remove_consecutive_duplicates(text):
    # split text into sentences
    sentences = text.split('. ')
    # remove consecutive duplicate sentences
    sentences = [sentences[i] for i in range(len(sentences)) if i == 0 or sentences[i] != sentences[i-1]]
    # rejoin sentences with a period separator
    text = '. '.join(sentences)
    # split text into words
    words = text.split()
    # remove consecutive duplicate words
    words = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i-1]]
    # rejoin words with a space separator
    text = ' '.join(words)
    return text

def remove_repeated_chars(string):
    new_string = ''
    count = 0
    prev_char = ''
    for char in string:
        if char == prev_char:
            count += 1
        else:
            count = 1
        if count <= 3:
            new_string += char
        elif count == 4:
            new_string = new_string[:-3]
        prev_char = char
    return new_string

@app.post("/transcribe2")
async def parse_input(input_arg:transCribe):
    start = time.perf_counter()     
    out_put={"Connection_Id":input_arg.connection_id,
             "Meeting_Id":input_arg.meeting_id,
             "lang":input_arg.extra_key,
             "speakerName":input_arg.speakerName,
             "isHost":input_arg.isHost,
             "deviceid":input_arg.device_id,
             "Cc_State":input_arg.Cc_State,
             "isFinal":input_arg.isFinal,
             "speakerLang":input_arg.extra_key,
             }

    decoded = base64.b64decode(input_arg.voice)
    audio_data = sr.AudioData(decoded, 16000,2)
    wav_data = io.BytesIO(audio_data.get_wav_data())
    audio_clip = AudioSegment.from_file(wav_data)

    # audio_clip = audio_clip.set_frame_rate(16000)
    # audio_clip = audio_clip.set_channels(1)

    audio_data=np.frombuffer(audio_clip.get_array_of_samples(), np.int16).flatten().astype(np.float32) / 32768.0
    # newsound = np.frombuffer(decoded, np.int16)
    # audio_data = torch.from_numpy(np.frombuffer(newsound, np.int16).flatten().astype(np.float32) / 32768.0)
    my_array = np.array(audio_data)
    vector_norm = np.linalg.norm(my_array)
    print(vector_norm)
    if vector_norm >10:
        result_segments, audio_info = model.transcribe(audio_data,
                                                    condition_on_previous_text=False,
                                                    language = input_arg.extra_key,
                                                    compression_ratio_threshold=2.4,
                                                    temperature=0.5,
                                                    without_timestamps=True,
                                                    vad_filter=True, 
                                                    vad_parameters=dict(threshold=0.5, 
                                                                        max_speech_duration_s=5,
                                                                        min_silence_duration_ms=500),
                                                    beam_size=1,
                                                    initial_prompt="",
                                                    log_prob_threshold=-1.00,
                                                    no_speech_threshold=0.6,
                                                    )
                                                    
        print("Detected language '%s' with probability %f" % (audio_info.language, audio_info.language_probability))
        Final_Text = " ".join([segment.text for segment in result_segments])
        # print(Final_Text,"..........0000000000")
        # if audio_info.language != input_arg.extra_key:
        #         return ""
        if not Final_Text.lower() in blacklist:
            processed_text = remove_consecutive_duplicates(Final_Text)
    
            s1 = processed_text
            while re.search(r'\b(.+)(\s+\1\b)+', s1):
                s1 = re.sub(r'\b(.+)(\s+\1\b)+', r'\1', s1)

            Final_Text = remove_repeated_chars(s1)
            dd={value['deviceid']: value['cc_lang']for value in input_arg.devicesInfo.values()}
        
            out_put.update({"text":Final_Text,"Language":dd})   
            das={input_arg.extra_key:Final_Text}
            did1 = set(dd.values())                        
            if input_arg.extra_key in did1:
                did1.remove(input_arg.extra_key )
            if len(did1) > 0:
                data_list = [{'text1': Final_Text, 'source1': input_arg.extra_key, 'target1': i} for i in did1]
                tasks = [asyncio.create_task(trans(x, das)) for x in data_list]
                await asyncio.wait(tasks)
            out_put["translation"] ={i:das.get(dd.get(i)) for i in dd}              
            rc.rpush("video_cc",json.dumps(out_put)) 
            # rc.rpush("transcript",json.dumps(out_put))   
            inference_time = time.perf_counter()-start
            print(f"{inference_time:.3f}s\t{Final_Text}")
            return Final_Text
if __name__ == "__main__":
   uvicorn.run(app, host="192.168.13.51", port=8000)

