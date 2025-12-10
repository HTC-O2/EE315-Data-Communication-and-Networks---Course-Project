import time
import threading
import random
import zlib  # 用于CRC校验
from enum import Enum
from cable import Cable  # 假设目录下有 cable.py，如果没有请使用之前的 mock

# ==========================================
# 0. 基础工具 (复用)
# ==========================================
def int_to_bits(n, l): return [int(b) for b in f"{n:0{l}b}"]
def bits_to_int(b): return int(''.join(map(str, b)), 2)
def string_to_bits(s): 
    return [int(b) for c in s for b in f"{ord(c):08b}"]
def bits_to_string(b):
    chars = []
    for i in range(0, len(b), 8):
        byte = b[i:i+8]
        if len(byte)==8: chars.append(chr(int(''.join(map(str, byte)), 2)))
    return "".join(chars)

# ==========================================
# 1. 协议定义 (Protocol Definition)
# ==========================================
class PacketType(Enum):
    DATA = 0
    ACK = 1

class ProtocolFrame:
    """
    帧结构: [Src(8)][Dst(8)][Seq(8)][Type(8)][Len(16)] + [Payload] + [CRC(32)]
    """
    @staticmethod
    def pack(src_addr, dst_addr, seq, ptype, payload_bits):
        # 构建头部
        header = (int_to_bits(src_addr, 8) + 
                  int_to_bits(dst_addr, 8) + 
                  int_to_bits(seq % 256, 8) +  # Requirement: Sequence Numbers
                  int_to_bits(ptype.value, 8) + 
                  int_to_bits(len(payload_bits), 16))
        
        content = header + payload_bits
        
        # 计算 CRC (Requirement: Reliable Transport - Integrity Check)
        data_bytes = bytearray()
        for i in range(0, len(content), 8):
            chunk = content[i:i+8]
            if len(chunk) < 8: chunk += [0]*(8-len(chunk))
            data_bytes.append(int(''.join(map(str, chunk)), 2))
        crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
        
        return content + int_to_bits(crc, 32)

    @staticmethod
    def unpack(bits):
        if len(bits) < 80: return None
        content = bits[:-32]
        received_crc = bits[-32:]
        
        # 校验 CRC
        data_bytes = bytearray()
        for i in range(0, len(content), 8):
            chunk = content[i:i+8]
            if len(chunk) < 8: chunk += [0]*(8-len(chunk))
            data_bytes.append(int(''.join(map(str, chunk)), 2))
        calc_crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
        
        if int_to_bits(calc_crc, 32) != received_crc:
            return None # 丢弃损坏包

        return {
            "src": bits_to_int(bits[0:8]),
            "dst": bits_to_int(bits[8:16]),
            "seq": bits_to_int(bits[16:24]), # Extraction of Sequence Number
            "type": PacketType(bits_to_int(bits[24:32])),
            "len": bits_to_int(bits[32:48]),
            "payload": bits[48:48+bits_to_int(bits[32:48])]
        }

# ==========================================
# 2. 传输层主机 (Transport Layer Implementation)
# ==========================================
class TransportHost:
    def __init__(self, address, cable):
        self.address = address
        self.cable = cable
        self.cable_lock = threading.Lock()
        
        # --- 状态变量 ---
        self.current_seq = 0            # 发送方: 当前序列号
        self.received_seqs = {}         # 接收方: 记录来自不同源的Seq (用于流控/去重)
        
        self.ack_event = threading.Event() # 用于等待ACK
        self.ack_received_seq = -1         # 收到的ACK中携带的Seq

        # 启动接收线程
        threading.Thread(target=self._listen_loop, daemon=True).start()

    # --- 物理层模拟 (不做修改) ---
    def _phy_send(self, bits):
        # 简单的 ASK 调制模拟
        signal = np.array([1.0 if b else 0.0 for b in bits for _ in range(5)], dtype=np.float32)
        with self.cable_lock:
            self.cable.transmit(signal)

    def _listen_loop(self):
        last_sig_time = 0
        while True:
            # 轮询获取信号 (模拟)
            if self.cable.last_output_signal is not None:
                # 简单的去重逻辑，防止同一信号读多次
                if time.time() - last_sig_time > 0.1: 
                    last_sig_time = time.time()
                    self._phy_receive(self.cable.last_output_signal)
            time.sleep(0.05)

    def _phy_receive(self, signal):
        # 解调
        bits = [1 if np.mean(signal[i:i+5]) > 0.5 else 0 for i in range(0, len(signal), 5)]
        self._transport_layer_receive(bits)

    # ==========================================================
    # 核心实现区域: 满足 Transport Layer 4项要求
    # ==========================================================

    def send_data(self, target_id, message):
        """
        可靠发送函数 (Reliable Send)
        """
        payload = string_to_bits(message)
        max_retries = 3
        timeout_sec = 2.0
        
        print(f"\n[Sender {self.address}] 开始发送数据: '{message}' -> Host {target_id}")
        
        # Requirement 2: Implement Sequence Numbers
        # 为当前包分配序列号
        seq = self.current_seq
        print(f"[Sender {self.address}] 分配序列号 Seq={seq}")

        for attempt in range(max_retries):
            # 1. 封装并发送数据帧
            frame = ProtocolFrame.pack(self.address, target_id, seq, PacketType.DATA, payload)
            self._phy_send(frame)
            print(f"[Sender {self.address}] 数据帧已发送 (Seq={seq}, 尝试 {attempt+1}/{max_retries})")

            # Requirement 3: Implement Timeout Retransmission
            # 2. 等待 ACK (阻塞，直到超时)
            self.ack_event.clear()
            if self.ack_event.wait(timeout=timeout_sec):
                # 收到信号，检查是否是对应的 Seq
                if self.ack_received_seq == seq:
                    # Requirement 1: Implement Reliable Transport (ACK received)
                    print(f"✅ [Sender {self.address}] 收到有效 ACK (Seq={seq})。发送成功。")
                    
                    # 更新序列号，准备发下一个包
                    self.current_seq = (self.current_seq + 1) % 256
                    return True
                else:
                    print(f"[Sender {self.address}] 收到 ACK 但序列号不匹配 (Exp:{seq}, Got:{self.ack_received_seq})")
            else:
                # 超时逻辑
                print(f"⚠️ [Sender {self.address}] 等待 ACK 超时 ({timeout_sec}s)！准备重传...")

        print(f"❌ [Sender {self.address}] 发送失败：达到最大重传次数。")
        return False

    def _transport_layer_receive(self, bits):
        """
        接收处理逻辑
        """
        packet = ProtocolFrame.unpack(bits)
        if not packet: return # CRC 校验失败或包结构错误

        dest = packet['dst']
        if dest != self.address: return # 不是发给我的

        src = packet['src']
        seq = packet['seq']
        ptype = packet['type']

        # --- 处理 ACK 包 ---
        if ptype == PacketType.ACK:
            print(f"[Receiver {self.address}] 收到 ACK 帧 (Seq={seq}) 来自 Host {src}")
            self.ack_received_seq = seq
            self.ack_event.set() # 通知发送线程
            return

        # --- 处理 DATA 包 ---
        if ptype == PacketType.DATA:
            print(f"[Receiver {self.address}] 收到 DATA 帧 (Seq={seq}) 来自 Host {src}")

            # Requirement 4: Implement Flow Control (Stop-and-Wait & De-duplication)
            # 检查是否是重复包
            last_seq = self.received_seqs.get(src, -1)
            
            if seq == last_seq:
                print(f"🛑 [Receiver {self.address}] 检测到重复包 (Seq={seq})，这是重传包。")
                print(f"   -> 操作: 丢弃数据，但重发 ACK 以防发送方没收到。")
                self._send_ack(src, seq)
                return
            
            # 如果是新包
            self.received_seqs[src] = seq # 更新状态
            msg = bits_to_string(packet['payload'])
            print(f"📩 [Receiver {self.address}] 数据有效，交付应用层: '{msg}'")
            
            # Requirement 1: Implement Reliable Transport (Send ACK)
            self._send_ack(src, seq)

    def _send_ack(self, target_id, seq):
        """发送确认帧"""
        # ACK 包不需要 Payload
        frame = ProtocolFrame.pack(self.address, target_id, seq, PacketType.ACK, [])
        self._phy_send(frame)
        print(f"[Receiver {self.address}] 发送 ACK (Seq={seq}) -> Host {target_id}")

# ==========================================
# 3. 验证测试 (Verification)
# ==========================================
import numpy as np

if __name__ == "__main__":
    # 模拟环境设置
    class MockCable(Cable):
        def transmit(self, signal):
            # 简单的广播模拟
            self.last_output_signal = signal

    # 1. 正常通信测试
    print("="*60)
    print("TEST 1: 正常通信 (验证 ACK, Seq, Stop-and-Wait)")
    print("="*60)
    cable = MockCable()
    host_a = TransportHost(10, cable)
    host_b = TransportHost(20, cable)

    # 启动 A 发送给 B
    threading.Thread(target=lambda: host_a.send_data(20, "Hello")).start()
    
    time.sleep(3) # 等待完成

    # 2. 模拟丢包/超时重传测试
    print("\n" + "="*60)
    print("TEST 2: 模拟丢包 (验证 Timeout Retransmission & Flow Control)")
    print("="*60)
    
    # 定义一个会“吃掉” ACK 的坏电缆
    class BrokenCable(MockCable):
        def __init__(self):
            super().__init__()
            self.packet_count = 0
            
        def transmit(self, signal):
            self.packet_count += 1
            # 策略: 丢弃第 2 个包 (即第一次发送的 ACK)
            # 第1个包是DATA(A->B)，第2个是ACK(B->A)，我们丢掉ACK
            if self.packet_count == 2:
                print("⚡ [Cable] 模拟网络故障: ACK 包丢失中途...")
                self.last_output_signal = None 
            else:
                self.last_output_signal = signal

    broken_cable = BrokenCable()
    host_c = TransportHost(30, broken_cable)
    host_d = TransportHost(40, broken_cable)
    
    host_c.send_data(40, "Packet with Lost ACK")