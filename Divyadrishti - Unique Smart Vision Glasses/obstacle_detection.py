import RPi.GPIO as GPIO
import time
import pyttsx3

GPIO.setmode(GPIO.BCM)
TRIG = 23
ECHO = 24
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

engine = pyttsx3.init()

def distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    start, stop = time.time(), time.time()
    while GPIO.input(ECHO) == 0:
        start = time.time()
    while GPIO.input(ECHO) == 1:
        stop = time.time()
    return (stop - start) * 17150

try:
    while True:
        dist = distance()
        print(f"Measured Distance = {dist:.1f} cm")
        if dist < 30:
            engine.say("Obstacle ahead")
            engine.runAndWait()
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
