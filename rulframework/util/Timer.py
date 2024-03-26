import threading
import time


class Timer:
    def __init__(self, keep_result: bool = True):
        self.keep_result = keep_result  # 保留计时结果
        self.__start_time = None
        self.__running = False
        self.__timer_thread = None

    def start(self):
        self.__start_time = time.time()
        self.__running = True
        self.__timer_thread = threading.Thread(target=self.__count)
        self.__timer_thread.start()

    def stop(self):
        self.__running = False
        self.__timer_thread.join()

    def __count(self):
        while self.__running:
            time.sleep(0.1)
            print(f'计时器：{round(time.time() - self.__start_time, 2)} s', end='\r')
        if self.keep_result:
            print(f'计时时长：{round(time.time() - self.__start_time, 2)} s')
