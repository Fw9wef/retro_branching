import threading
from threading import Thread
import queue as qq
from queue import Queue


def func(queue):
    curr_thread = threading.current_thread()
    while True:
        try:
            queue_value = queue.get(timeout=1)
        except qq.Empty:
            break
        print("thread %s, queue value %d" % (curr_thread, queue_value))
        queue.task_done()
    print("thread %s finished" % curr_thread)


def main():
    queue = Queue(maxsize=5)
    threads = list()
    for _ in range(2):
        process = Thread(target=func, args=(queue,))
        process.start()
        threads.append(process)

    for i in range(10):
        queue.put(i)
        print("main putted %d" % i)

    for process in threads:
        process.join()
        print("join")

    queue.join()
    print("join")

    print("all done")


if __name__ == "__main__":
    main()
