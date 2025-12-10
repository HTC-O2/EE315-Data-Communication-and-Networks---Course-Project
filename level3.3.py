import time
import threading
import json  # 用于模拟复杂的数据体格式
from enum import Enum

# ==========================================
# 0. 评分验证打印工具
# ==========================================


def log_process(msg):
    print(f"   [AppProcess] {msg}")

# ==========================================
# 1. 协议解析器 (Protocol Parsing)
# 对应要求: Implement protocol parsing
# ==========================================
class AppLayerProtocol:
    """
    定义一个简单的文本协议 (SimpleHTTP)
    格式：
      Request:  METHOD|URI|BODY
      Response: STATUS_CODE|STATUS_TEXT|BODY
    """
    @staticmethod
    def build_request(method, uri, body=""):
        # 构建请求字符串
        raw = f"{method}|{uri}|{body}"
        return raw
    @staticmethod
    def build_response(status_code, status_text, body=""):
        # 构建响应字符串
        raw = f"{status_code}|{status_text}|{body}"
        return raw
    @staticmethod
    def parse(raw_data):
        """
        解析收到的字符串，判断是请求还是响应
        """
        try:
            parts = raw_data.split('|', 2)
            if len(parts) < 3: return None
            
            p1, p2, p3 = parts[0], parts[1], parts[2]
            # 判断逻辑：如果第一部分是数字，认为是 Response (状态码)
            # 如果是单词 (GET/POST)，认为是 Request
            if p1.isdigit():
                # 是响应 Response
                log_process(f"解析响应: Code={p1}, Text={p2}")

                return {
                    "type": "RESPONSE",
                    "status": int(p1),
                    "text": p2,
                    "body": p3
                }
            else:
                # 是请求 Request
                log_process(f"解析请求: Method={p1}, URI={p2}")

                return {
                    "type": "REQUEST",
                    "method": p1,
                    "uri": p2,
                    "body": p3
                }
        except Exception as e:
            print(f"解析错误: {e}")
            return None

# ==========================================
# 2. 模拟底层传输 (Mock Transport)
# 这一部分是为了让代码能跑起来，模拟传输层的收发
# ==========================================
class MockTransport:
    def __init__(self):
        self.peers = {} # address -> handler

    def register(self, addr, handler):
        self.peers[addr] = handler

    def send(self, src, dst, data):
        # 模拟网络延迟
        time.sleep(0.1)
        if dst in self.peers:
            # 直接调用对方的回调
            threading.Thread(target=self.peers[dst], args=(src, data)).start()
        else:
            print(f"Error: {dst} not found")

# ==========================================
# 3. 服务端设计 (Server Side)
# 对应要求: Design application layer protocol
# ==========================================
class SimpleServer:
    def __init__(self, address, transport):
        self.address = address
        self.transport = transport
        self.transport.register(address, self.on_receive)
        # 简单的路由表
        self.routes = {
            "/index": "Welcome to SimpleServer Home",
            "/time": lambda: f"ServerTime: {time.time()}",
            "/api/data": "JSON_DATA_Here"
        }
        print(f"服务器 [{self.address}] 已启动，正在监听...")

    def on_receive(self, src_addr, raw_data):
        # 1. 解析协议
        msg = AppLayerProtocol.parse(raw_data)
        
        if msg and msg['type'] == 'REQUEST':
            self.handle_request(src_addr, msg)

    def handle_request(self, client_addr, req):
        log_process(f"服务器处理请求: {req['method']} {req['uri']}")
        
        method = req['method']
        uri = req['uri']
        
        # 业务逻辑处理
        response_body = ""
        status_code = 200
        status_text = "OK"

        if uri in self.routes:
            handler = self.routes[uri]
            if callable(handler):
                response_body = handler()
            else:
                response_body = handler
        else:
            status_code = 404
            status_text = "Not Found"
            response_body = "The requested resource does not exist."

        # 2. 构建响应
        resp_str = AppLayerProtocol.build_response(status_code, status_text, response_body)
        
        # 3. 发送回客户端
        self.transport.send(self.address, client_addr, resp_str)


# ==========================================
# 4. 客户端设计 (Client Side)
# 对应要求: Implement request-response pattern
# ==========================================
class SimpleClient:
    def __init__(self, address, transport):
        self.address = address
        self.transport = transport
        self.transport.register(address, self.on_receive)
        # 用于实现同步的 Request-Response 模式
        self.pending_response = None
        self.response_event = threading.Event()
    def on_receive(self, src_addr, raw_data):
        # 收到数据，解析
        msg = AppLayerProtocol.parse(raw_data)
        if msg and msg['type'] == 'RESPONSE':
            self.pending_response = msg
            # 解除阻塞，通知主线程响应已到达
            self.response_event.set()

    def get(self, target_server, uri):
        """
        发送 GET 请求并等待响应 (同步阻塞模式)
        """
        print(f"\n[Client] 发起 GET 请求 -> {target_server}:{uri}")
        
        # 1. 构建请求
        req_str = AppLayerProtocol.build_request("GET", uri)
        
        # 2. 重置事件
        self.response_event.clear()
        self.pending_response = None
        
        # 3. 发送
        self.transport.send(self.address, target_server, req_str)
        
        # 4. 等待响应 (Request-Response Pattern)
        log_process("客户端正在等待服务端响应...")
        if self.response_event.wait(timeout=2.0):

            return self.pending_response
        else:
            print("Error: 请求超时")
            return None

# ==========================================
# 5. 验证测试
# ==========================================
if __name__ == "__main__":
    # 搭建环境
    transport = MockTransport()
    server = SimpleServer("192.168.1.1", transport)
    client = SimpleClient("192.168.1.100", transport)

    print("\n" + "="*60)
    print("TEST START: Application Layer Protocol")
    print("="*60)

    # --- 测试 1: 请求存在的资源 (200 OK) ---
    print("\n>>> Case 1: 访问 /index")
    response = client.get("192.168.1.1", "/index")
    
    if response:
        print(f"   最终结果: [{response['status']} {response['text']}] Body: {response['body']}")
    
    # --- 测试 2: 访问动态资源 ---
    print("\n>>> Case 2: 访问 /time (动态生成)")
    response = client.get("192.168.1.1", "/time")
    if response:
        print(f"   最终结果: Body: {response['body']}")

    # --- 测试 3: 访问不存在的资源 (404 Not Found) ---
    print("\n>>> Case 3: 访问 /unknown (测试错误处理)")
    response = client.get("192.168.1.1", "/unknown")
    if response:
        print(f"   最终结果: [{response['status']} {response['text']}]")

    print("\n" + "="*60)
    print("TEST END")
    print("="*60)