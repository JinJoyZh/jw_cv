#!/usr/bin/env python
import json
import threading
import uuid

import pika

HOST_IP         = '172.17.0.3'
EXCHANGE        = 'zjsxyjy'
EXCHANGE_TYPE   = 'topic'
BINDING_KEY     = 'alg.monitor'

class FibonacciRpcClient(object):

    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=HOST_IP))

        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=EXCHANGE, exchange_type=EXCHANGE_TYPE)

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        print(f'__init__ $self.callback_queue {self.callback_queue}')
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, n):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=str(n))
        self.connection.process_data_events(time_limit=None)
        return self.response
    
    def subscribe_alg_output(self, callback):
        
        result = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue
        print(f'subscribe_alg_output$queue_name{queue_name}')
        self.channel.queue_bind(
             exchange=EXCHANGE, queue=queue_name, routing_key=BINDING_KEY)
        print(' [*] Waiting for alg outputs. To exit press CTRL+C')
        self.channel.basic_consume(
        queue=queue_name, on_message_callback=callback, auto_ack=True)
        self.channel.start_consuming()

if __name__ == '__main__':
    fibonacci_rpc = FibonacciRpcClient()

    def callback(ch, method, properties, body):
            #算法返回的数据在body中
            print(body)
    
    task = threading.Thread(target=fibonacci_rpc.subscribe_alg_output, args=[callback])
    task.start()

    fibonacci_rpc2 = FibonacciRpcClient()

    def foo():
        msg = {
            "input_path":"/workspace/input",
            "output_path":"/workspace/output"
        }
        msg = json.dumps(msg)
        print(f"Requesting {msg}")
        response = fibonacci_rpc2.call(msg)
        print(f" [.] Got {response}")    
    task1 = threading.Thread(target=foo)
    task1.start()
