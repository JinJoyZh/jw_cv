# #!/usr/bin/env python
# import json
# import pika
# from main import get_defog_vide, get_keyframes
# from track import run_track
# from concurrent.futures import ThreadPoolExecutor


# HOST_IP         = '172.17.0.3'
# EXCHANGE        = 'zjsxyjy'
# EXCHANGE_TYPE   = 'topic'
# BINDING_KEY     = 'alg.monitor'

# FFMPEG_PATH = ""
# IS_RTSP_PIPE_ON = True

# connection_emit = pika.BlockingConnection(pika.ConnectionParameters(host=HOST_IP))
# channel_emit = connection_emit.channel()
# channel_emit.exchange_declare(exchange=EXCHANGE, exchange_type=EXCHANGE_TYPE)

# thread_pool = ThreadPoolExecutor(max_workers=12)

# def emit_frame_prediction(prediction):
#     print(f"emit {prediction}")
#     channel_emit.basic_publish(
#         exchange=EXCHANGE , routing_key=BINDING_KEY, body=prediction)
#     print(f" [x] Sent {BINDING_KEY}:{prediction}")

# def exec_task(msg):
#     session_type = msg['session_type']
#     if session_type == 'exec_alg':
#         thread_pool.submit(run_track, msg, emit_frame_prediction)
#     if session_type == 'get_keyframes':
#         get_keyframes(msg)
#     if session_type == 'get_defog_vide':
#         get_defog_vide(msg)
#     return 1

# def on_request(ch, method, props, body):
#     msg = body.decode()
#     msg = json.loads(msg)
#     print(f" [x] {method.routing_key}:{msg}")
#     response = exec_task(msg)
#     ch.basic_publish(exchange='',
#                     routing_key=props.reply_to,
#                     properties=pika.BasicProperties(correlation_id= \
#                                                             props.correlation_id),
#                     body=str(response))
#     ch.basic_ack(delivery_tag=method.delivery_tag)

# connection_rpc = pika.BlockingConnection(pika.ConnectionParameters(host=HOST_IP))
# channel_rpc = connection_rpc.channel()
# channel_rpc.basic_qos(prefetch_count=1)
# channel_rpc.queue_declare(queue='rpc_queue')
# channel_rpc.basic_consume(queue='rpc_queue', on_message_callback=on_request)
# print(" [x] Awaiting RPC requests")
# channel_rpc.start_consuming()


