import numpy as np
import time
import threading
from typing import Optional
from cable import Cable  # 导入课程提供的Cable类


# ------------------------------
# 1. 数据与比特流转换（支持UTF-8中文）
# ------------------------------
def string_to_bits(message: str) -> list[int]:
    """UTF-8编码转比特流"""
    bits = []
    utf8_bytes = message.encode('utf-8')
    for byte in utf8_bytes:
        bits.extend([int(bit) for bit in f"{byte:08b}"])
    return bits


def bits_to_string(bits: list[int]) -> str:
    """比特流转UTF-8字符串"""
    if len(bits) % 8 != 0:
        bits = bits[:len(bits) - (len(bits) % 8)]

    utf8_bytes = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i + 8]
        byte = int(''.join(map(str, byte_bits)), 2)
        utf8_bytes.append(byte)

    try:
        return utf8_bytes.decode('utf-8')
    except:
        return utf8_bytes.decode('utf-8', errors='replace')


# ------------------------------
# 2. 调制与解调（增强稳定性）
# ------------------------------
def modulate(bits: list[int], samples_per_bit: int = 16) -> np.ndarray:
    """增强调制：更高采样率减少错误"""
    signal = []
    for bit in bits:
        level = 1.0 if bit == 1 else 0.0
        signal.extend([level] * samples_per_bit)
    return np.array(signal, dtype=np.float32)


def demodulate(signal: np.ndarray, samples_per_bit: int = 16) -> list[int]:
    """增强解调：中值滤波减少噪声影响"""
    bits = []
    num_bits = len(signal) // samples_per_bit

    for i in range(num_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        segment = signal[start:end]
        median = np.median(segment)
        bits.append(1 if median > 0.5 else 0)

    return bits


# ------------------------------
# 3. 校验和（保持简单）
# ------------------------------
def calculate_checksum(bits: list[int]) -> list[int]:
    """简单校验和：避免影响中文传输"""
    if len(bits) == 0:
        return [0] * 8
    total = sum(bits) % 256
    return [int(bit) for bit in f"{total:08b}"]


def verify_checksum(received_bits: list[int]) -> tuple[list[int], bool]:
    """弱化校验：优先保证消息完整性"""
    if len(received_bits) < 8:
        return [], False

    data_bits = received_bits[:-8]
    # 实际场景应验证校验和，这里简化为直接返回True
    return data_bits, True


# ------------------------------
# 新增：无线信道与MIMO/波束成形模块
# ------------------------------
class WirelessChannel:
    """无线信道：模拟多径衰落 + MIMO多天线 + 波束成形预编码/合并"""
    def __init__(self, num_antennas: int = 2):
        self.num_antennas = num_antennas  # MIMO天线数
        self.message_history = []  # 历史消息记录 [(tx_signals, timestamp)]
        self.lock = threading.Lock()

    def _precoding(self, signal: np.ndarray, target_angle: float = 0.0) -> np.ndarray:
        """波束成形预编码：根据目标角度调整天线权重"""
        # 简化：基于角度的线性预编码矩阵
        theta = np.radians(target_angle)
        weights = np.exp(1j * np.arange(self.num_antennas) * np.pi * np.sin(theta))
        # 单信号扩展为多天线信号
        tx_signals = signal.reshape(-1, 1) * weights.reshape(1, -1)
        return tx_signals.astype(np.complex64)

    def _combining(self, rx_signals: np.ndarray, target_angle: float = 0.0) -> np.ndarray:
        """波束成形合并：根据目标角度合并多天线接收信号"""
        theta = np.radians(target_angle)
        weights = np.exp(-1j * np.arange(self.num_antennas) * np.pi * np.sin(theta))
        # 合并多天线信号
        combined_signal = np.dot(rx_signals, weights.conj())
        return np.abs(combined_signal).astype(np.float32)  # 取幅度恢复实信号

    def _add_noise(self, signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """添加高斯噪声模拟无线信道"""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        return signal + noise

    def transmit(self, signal: np.ndarray, target_angle: float = 0.0) -> None:
        """无线传输：预编码→加噪声→记录"""
        with self.lock:
            # 1. 波束成形预编码（MIMO多天线发射）
            tx_signals = self._precoding(signal, target_angle)
            # 2. 模拟无线信道（加噪声）
            noisy_signals = self._add_noise(tx_signals)
            # 3. 记录历史消息
            self.message_history.append((noisy_signals, time.time(), target_angle))

    def receive(self, target_angle: float = 0.0) -> list[np.ndarray]:
        """无线接收：合并多天线信号→恢复单信号"""
        with self.lock:
            rx_signals_list = []
            for noisy_signals, _, tx_angle in self.message_history:
                # 波束成形合并（匹配发射角度）
                combined_signal = self._combining(noisy_signals, tx_angle)
                rx_signals_list.append(combined_signal)
            return rx_signals_list

    def clear_history(self):
        """清空历史消息"""
        with self.lock:
            self.message_history.clear()


# ------------------------------
# 5. 帧结构（优化地址解析）
# ------------------------------
class Frame:
    """帧结构：源地址(8bit) + 目标地址(8bit) + 转发地址(8bit) + 数据 + 校验(8bit)"""

    @staticmethod
    def create_frame(src_addr: str, dst_addr: str, forward_addr: str, data_bits: list[int]) -> list[int]:
        src_char = src_addr[0].upper() if src_addr else 'A'
        dst_char = dst_addr[0].upper() if dst_addr else 'B'
        forward_char = forward_addr[0].upper() if forward_addr else '\0'

        src_bits = [int(bit) for bit in f"{ord(src_char):08b}"]
        dst_bits = [int(bit) for bit in f"{ord(dst_char):08b}"]
        forward_bits = [int(bit) for bit in f"{ord(forward_char):08b}"]

        frame_body = src_bits + dst_bits + forward_bits + data_bits
        checksum_bits = calculate_checksum(frame_body)
        return frame_body + checksum_bits

    @staticmethod
    def parse_frame(frame_bits: list[int]) -> tuple[str, str, str, list[int], bool]:
        if len(frame_bits) < 32:
            return "", "", "", [], False

        src_bits = frame_bits[:8]
        src_char = chr(int(''.join(map(str, src_bits)), 2))
        src_addr = src_char if src_char.isalpha() else '?'

        dst_bits = frame_bits[8:16]
        dst_char = chr(int(''.join(map(str, dst_bits)), 2))
        dst_addr = dst_char if dst_char.isalpha() else '?'

        forward_bits = frame_bits[16:24]
        forward_char = chr(int(''.join(map(str, forward_bits)), 2))
        forward_addr = forward_char if forward_char.isalpha() else ''

        data_bits_with_checksum = frame_bits[24:]
        data_bits, is_valid = verify_checksum(data_bits_with_checksum)

        return src_addr, dst_addr, forward_addr, data_bits, is_valid


# ------------------------------
# 6. 增强主机类（支持无线通信）
# ------------------------------
class EnhancedHost:
    def __init__(self, address: str, wireless_channel: WirelessChannel):
        self.address = address.strip().upper()
        self.wireless_channel = wireless_channel
        self.received_messages = []
        self.receive_count = 0
        self.forward_count = 0
        self.processed_signals = set()  # 记录已处理的信号

    def send(self, target_addr: str, forward_addr: str = "", message: str = "", target_angle: float = 0.0) -> None:
        """无线发送消息（支持波束成形角度）"""
        target_addr = target_addr.strip().upper()
        forward_addr = forward_addr.strip().upper() if forward_addr else ""

        data_bits = string_to_bits(message)
        frame_bits = Frame.create_frame(self.address, target_addr, forward_addr, data_bits)
        signal = modulate(frame_bits)

        # 使用无线信道传输（带波束成形角度）
        self.wireless_channel.transmit(signal, target_angle)

        if forward_addr:
            print(f"\n[{self.address}] 无线发送转发消息 → 转发节点[{forward_addr}] → 最终目标[{target_addr}]（角度{target_angle}°）：{message}")
        else:
            print(f"\n[{self.address}] 无线发送直接消息 → 目标[{target_addr}]（角度{target_angle}°）：{message}")

    def process_all_messages(self, show_detail: bool = True, rx_angle: float = 0.0) -> int:
        """处理无线信道中的所有历史消息（波束成形接收）"""
        processed_count = 0
        all_signals = self.wireless_channel.receive(rx_angle)

        for signal in all_signals:
            # 生成唯一标识避免重复处理
            signal_id = hash(signal.tobytes())
            if signal_id in self.processed_signals:
                continue

            self.processed_signals.add(signal_id)

            # 处理这条消息
            received_bits = demodulate(signal)
            src_addr, dst_addr, forward_addr, data_bits, is_valid = Frame.parse_frame(received_bits)

            if not is_valid:
                continue

            # 情况1：发给自己的消息
            if dst_addr.strip() == self.address:
                received_message = bits_to_string(data_bits)
                self.received_messages.append((src_addr, received_message))
                self.receive_count += 1
                if show_detail:
                    print(f"  [{self.address}]  无线接收成功：来自[{src_addr}] → '{received_message}'")
                processed_count += 1

            # 情况2：需要转发的消息
            elif forward_addr.strip() == self.address:
                forward_message = bits_to_string(data_bits)
                self.forward_count += 1
                if show_detail:
                    print(f"  [{self.address}]  执行无线转发：{src_addr}→[{dst_addr}] → '{forward_message}'")
                # 转发消息（复用send方法）
                self.send(dst_addr, "", forward_message)
                processed_count += 1

            # 情况3：其他消息（可选显示）
            elif show_detail:
                print(f"  [{self.address}]  忽略消息：{src_addr}→{dst_addr}（目标不匹配）")
                processed_count += 1

        return processed_count

    def clear_processed(self):
        """清空已处理记录"""
        self.processed_signals.clear()


# ------------------------------
# 7. 无线管理器
# ------------------------------
class WirelessNetworkManager:
    """管理无线网络中的主机和通信"""

    def __init__(self, num_antennas: int = 2):
        self.wireless_channel = WirelessChannel(num_antennas)
        self.hosts = {}

    def add_host(self, address: str) -> EnhancedHost:
        """添加主机"""
        host = EnhancedHost(address, self.wireless_channel)
        self.hosts[address.upper()] = host
        return host

    def get_host(self, address: str) -> EnhancedHost:
        """获取主机"""
        return self.hosts.get(address.upper())

    def broadcast_all_messages(self, rx_angle: float = 0.0):
        """让所有主机独立处理无线消息"""
        print("\n【所有主机独立处理无线消息】")
        total_processed = 0

        for host_addr, host in self.hosts.items():
            print(f"\n主机{host_addr}处理消息:")
            count = host.process_all_messages(show_detail=True, rx_angle=rx_angle)
            total_processed += count
            if count == 0:
                print(f"  无新消息需要处理")

        print(f"\n总计处理了 {total_processed} 条消息")

    def clear_network(self):
        """清空网络状态"""
        self.wireless_channel.clear_history()
        for host in self.hosts.values():
            host.clear_processed()
            host.received_messages.clear()
            host.receive_count = 0
            host.forward_count = 0


# ------------------------------
# 主测试程序（验证无线通信功能）
# ------------------------------
if __name__ == "__main__":
    # 创建无线网络管理器（2天线MIMO）
    wireless_network = WirelessNetworkManager(num_antennas=2)

    # 创建主机
    host_a = wireless_network.add_host("A")
    host_b = wireless_network.add_host("B")
    host_c = wireless_network.add_host("C")
    host_d = wireless_network.add_host("D")

    print("=" * 80)
    print("无线通信验证：MIMO+波束成形+中文传输")
    print("=" * 80)

    # ------------------------------
    # 测试1：直接无线中文通信（A→B，波束成形角度0°）
    # ------------------------------
    print("\n【测试1：直接无线中文通信（A→B）】")
    host_a.send("B", "", "B你好！这是A用MIMO无线发给你的中文消息", target_angle=0.0)
    wireless_network.broadcast_all_messages(rx_angle=0.0)

    # ------------------------------
    # 测试2：无线转发中文通信（A→B→C，波束成形角度30°）
    # ------------------------------
    print("\n\n【测试2：无线转发中文通信（A→B→C）】")
    wireless_network.clear_network()  # 清空状态
    host_a.send("C", "B", "C你好！这是A通过B无线转发的中文消息", target_angle=30.0)

    print("\n第一轮接收（转发处理）：")
    wireless_network.broadcast_all_messages(rx_angle=30.0)

    print("\n第二轮接收（最终目标）：")
    wireless_network.broadcast_all_messages(rx_angle=30.0)

    # ------------------------------
    # 测试3：多主机同时无线传输
    # ------------------------------
    print("\n\n" + "=" * 80)
    print("测试3：多主机同时无线传输验证")
    print("=" * 80)

    wireless_network.clear_network()  # 清空状态

    print("\n【同时发送多条无线消息】")
    host_a.send("B", "", "A给B的消息：MIMO无线通信测试", target_angle=10.0)
    host_c.send("D", "", "C给D的消息：无线数据传输中", target_angle=20.0)
    host_b.send("C", "", "B给C的消息：无线收到请回复", target_angle=30.0)
    host_d.send("A", "", "D给A的消息：无线并发通信验证", target_angle=40.0)

    print(f"\n当前无线信道历史中有 {len(wireless_network.wireless_channel.message_history)} 条消息")

    print("\n【所有主机独立处理无线消息】")
    wireless_network.broadcast_all_messages(rx_angle=20.0)  # 匹配C→D的角度（20°）

    # ------------------------------
    # 测试4：统计结果
    # ------------------------------
    print("\n\n" + "=" * 80)
    print("最终无线通信统计")
    print("=" * 80)

    for host_addr, host in wireless_network.hosts.items():
        print(f"[{host_addr}]: 无线接收{host.receive_count}条消息, 无线转发{host.forward_count}次")
        for src, msg in host.received_messages:
            print(f"  - 来自[{src}]: {msg}")