import threading

# 多源状态配置 source_state_array
# [
#     {
#             "session_type":"exec_alg",
#             "alg_combination":"<string-required> 需要调用的算法名称的组合",
#             "source":"<string-required> 以rtsp://开头的url"
#             "width": 11,
#             "height":22,
#             "trackers":{
#                     "123":  {"tracker": tracker_1, "label": label_1"},
#                     "345":  {"tracker": tracker_2, "label": label_2"}
#             },
#             "adjust_boudingbox":[
#                         {
#                             "id":"<int - required> 目标id号",
#                             "scale_w":"<float - required> 宽度缩放比例",
#                             "scale_h":"<float - required> 高度缩放比例",
#                             "shift_x":"<int - required> x轴偏移量",
#                             "shift_y":"<int - required> y轴偏移量"
#                         },
#                         {
#                             "id":"<int - required> 目标id号",
#                             "scale_w":"<float - required> 宽度缩放比例",
#                             "scale_h":"<float - required> 高度缩放比例",
#                             "shift_x":"<int - required> x轴偏移量",
#                             "shift_y":"<int - required> y轴偏移量"
#                         }
#                     ]
#         }
# ]

def synchonized(func):
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kargs):
        with func.__lock__:
            return func(*args, **kargs)
    
    return lock_func

class SourceStatesKeeper(object):
    instance = None
    source_state_array = []

    @synchonized
    def __new__(cls, *args, **kargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

if __name__ == "__main__":
    a = SourceStatesKeeper()
    print(a.source_state_array)