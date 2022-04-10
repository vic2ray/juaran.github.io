---
title: 自制SYN扫描
date: 2021-03-25
category: 计算机网络
tag: TCP
---

* [Python3 socket模块官方文档](https://docs.python.org/zh-cn/3/library/socket.html#creating-sockets )
* [python3 struct模块 处理二进制 pack unpack用法](https://blog.csdn.net/whatday/article/details/100559721 )
* [python 使用raw socket进行TCP SYN扫描](https://blog.csdn.net/Jeanphorn/article/details/45226947 )
* [IP协议首部详细分析](https://blog.csdn.net/zhangdaisylove/article/details/47147991 )


<!-- more -->

大致过程是：根据IP首部格式，使用`pack`打包成二进制形式的数据报，在通过`socket`发送。

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/20150730120353079" alt="img" style="zoom: 60%;" />

在pack中打包数据到二进制的格式fmt为：

* B：代表byte一个字节，8个bit即8位
* H：代表2字节，16位
* s：代表一个字符，8位。4s即32位字符4字节32位。格式化对象需为字节类型

```python
# IP 首部构造
Version = 4        # IPv4，版本号
IHL = 5 # IP报文首部长度，20byte = 4byte * 5
Version_IHL = (Version << 4) + IHL    # 版本号在高四位，首部长度在低四位
TOS =  0    # Type of Service, 0代表一般服务
TL = 20 + 40     # IP报文总长度 = 头部20字节 + 可变数据字节数
Id = random.randint(18000, 65535) # 随机标识identification
FFO = 0        # 标志位flag+片偏移fragment offset均为0
TTL = 255    # Time to Live，能经过多少个路由器，最大255
protocol = 6    # TCP 协议号
checksum = 0    # 头部校验和，0不校验
source_addr = socket.inet_aton('x.x.x.x')        # 将ip字符串转为32位二进制格式
dest_addr = socket.inet_aton('x.x.x.x')        # 目的IP
 
IP_Header = pack('!BBHHHBBH4s4s', Version_IHL, TOS, TL, Id, FFO, TTL, protocol, checksum, source_addr, dest_addr)
```

IHL解释：

> IHL(Internet Header Length 报头长度)是计算机名词，位于IP报文的第二个字段，4位，表示IP报文头部按32位字长（32位，4字节）计数的长度，也即报文头的长度等于IHL的值乘以4。
> 
> 由于IPv4的头部为变长，所以需要用该字段来标示IP报文头的长度，也等同于数据字段的偏移量。最小为5，即5×32 = 160位 = 20字节。最大为15，表示15×32 bits = 480位 = 60字节。

* [Python socket编程之构造IP首部和ICMP首部](https://www.ktanx.com/blog/p/3082)
* [python 使用raw socket进行TCP SYN扫描](https://blog.csdn.net/Jeanphorn/article/details/45226947)
* [TCP报文段的首部格式 ](https://blog.csdn.net/qq_32998153/article/details/79680704)
* [TCP校验和（Checksum）的原理和实现 ](https://blog.csdn.net/qq_15437629/article/details/79183076)

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/20180324192146298" alt="img" style="zoom: 50%;" />

```python
# TCP 首部创建
source_port = random.randint(30000, 65535)  # 本地随机端口
dest_port = 3306
seq_number = 0            # 序号
ack_number = 0      # 确认号
header_length = 5         # 首部长，20字节=4字节*5
reserved = 0        # 保留
hl_resrv = (header_length << 4) + reserved  # 拼接字节
# tcp flags 标志位
urg = 0
ack = 0
psh = 0
rst = 0
syn = 1        # SYN
fin = 0
# 拼成0x002的SYN标志位
tcp_flags = fin + (syn<<1) + (rst<<2) + (psh<<3) + (ack<<4) + (urg<<5)
window = 8192    # 窗口
checksum = 0    # 待校验
urgent_pointer = 0    # 紧急指针
 
tcp_header = pack('!HHLLBBHHH', source_port, dest_port, seq_number, ack_number, hl_resrv, tcp_flags, window, checksum, urgent_pointer)
 
# 伪首部12字节
source_addr = socket.inet_aton(source_addr)
print(source_addr)
dest_addr = socket.inet_aton(dest_addr)  # 目的IP
zeros = 0
protocol = 6  # TCP 协议号
tcp_length = len(tcp_header)  # 首部字节数
pseudo_header = pack('!4s4sBBH', source_addr, dest_addr, zeros, protocol, tcp_length)
 
# 计算校验和
checksum = check_sum(pseudo_header + tcp_header)
 
# 重新打包
tcp_header = pack('!HHLLBBHHH', source_port, dest_port, seq_number, ack_number, hl_resrv, tcp_flags, window,
                  checksum, urgent_pointer)
```

计算校验和：可能是错误的

```python
def check_sum(msg):
    """
    计算TCP校验和
    :param msg:
    :return:
    """
    s = 0
    # 每次取2个字节 = 16位
    for i in range(0, len(msg), 2):
        w = (msg[i] << 8) + (msg[i+1])
        s = s+w
 
    s = (s >> 16) + (s & 0xffff)
    s = ~s & 0xffff
    return s
```

* 一个更为准确的构造：[https://gist.github.com/fffaraz/57144833c6ef8bd9d453](https://gist.github.com/fffaraz/57144833c6ef8bd9d453)
* [ python2.7和python3.4中的ord函数不同？](https://www.cnpython.com/qa/128974)
* [SYN数据包没有回复（Python）原始套接字](https://stackoverflow.com/questions/44380251/syn-packet-getting-no-replies-python-raw-sockets)

终于让我找到了！

### Creating a SYN port scanner

https://inc0x0.com/tcp-ip-packets-introduction/tcp-ip-packets-4-creating-a-syn-port-scanner/

> $ sudo python3 tcp_syn.py
> [sudo] june 的密码：
> Port 21 is: closed
> Port 22 is: closed
> Port 80 is: closed
> Port 8080 is: closed

```python
import socket
from struct import *
import binascii
 
class Packet:
    def __init__(self, src_ip, dest_ip, dest_port):
        # https://docs.python.org/3.7/library/struct.html#format-characters
        # all values need to be at least one byte long (-> we need to add up some values)
 
        ############
        # IP segment
        self.version = 0x4
        self.ihl = 0x5
        self.type_of_service = 0x0
        self.total_length = 0x28
        self.identification = 0xabcd
        self.flags = 0x0
        self.fragment_offset = 0x0
        self.ttl = 0x40
        self.protocol = 0x6
        self.header_checksum = 0x0
        self.src_ip = src_ip
        self.dest_ip = dest_ip
        self.src_addr = socket.inet_aton(src_ip)
        self.dest_addr = socket.inet_aton(dest_ip)
        self.v_ihl = (self.version << 4) + self.ihl
        self.f_fo = (self.flags << 13) + self.fragment_offset
 
        #############
        # TCP segment
        self.src_port = 0x3039
        self.dest_port = dest_port
        self.seq_no = 0x0
        self.ack_no = 0x0
        self.data_offset = 0x5
        self.reserved = 0x0
        self.ns, self.cwr, self.ece, self.urg, self.ack, self.psh, self.rst, self.syn, self.fin = 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0
        self.window_size = 0x7110
        self.checksum = 0x0
        self.urg_pointer = 0x0
        self.data_offset_res_flags = (self.data_offset << 12) + (self.reserved << 9) + (self.ns << 8) + (
                    self.cwr << 7) + (self.ece << 6) + (self.urg << 5) + (self.ack << 4) + (self.psh << 3) + (
                                                 self.rst << 2) + (self.syn << 1) + self.fin
 
        ########
        # packet
        self.tcp_header = b""
        self.ip_header = b""
        self.packet = b""
 
    def calc_checksum(self, msg):
        s = 0
        for i in range(0, len(msg), 2):
            w = (msg[i] << 8) + msg[i + 1]
            s = s + w
        # s = 0x119cc
        s = (s >> 16) + (s & 0xffff)
        # s = 0x19cd
        s = ~s & 0xffff
        # s = 0xe632
        return s
 
    def generate_tmp_ip_header(self):
        tmp_ip_header = pack("!BBHHHBBH4s4s", self.v_ihl, self.type_of_service, self.total_length,
                             self.identification, self.f_fo,
                             self.ttl, self.protocol, self.header_checksum,
                             self.src_addr,
                             self.dest_addr)
        return tmp_ip_header
 
    def generate_tmp_tcp_header(self):
        tmp_tcp_header = pack("!HHLLHHHH", self.src_port, self.dest_port,
                              self.seq_no,
                              self.ack_no,
                              self.data_offset_res_flags, self.window_size,
                              self.checksum, self.urg_pointer)
        return tmp_tcp_header
 
    def generate_packet(self):
        # IP header + checksum
        final_ip_header = pack("!BBHHHBBH4s4s", self.v_ihl, self.type_of_service, self.total_length,
                               self.identification, self.f_fo,
                               self.ttl, self.protocol, self.calc_checksum(self.generate_tmp_ip_header()),
                               self.src_addr,
                               self.dest_addr)
        # TCP header + checksum
        tmp_tcp_header = self.generate_tmp_tcp_header()
        pseudo_header = pack("!4s4sBBH", self.src_addr, self.dest_addr, self.checksum, self.protocol,
                             len(tmp_tcp_header))
        psh = pseudo_header + tmp_tcp_header
        final_tcp_header = pack("!HHLLHHHH", self.src_port, self.dest_port,
                                self.seq_no,
                                self.ack_no,
                                self.data_offset_res_flags, self.window_size,
                                self.calc_checksum(psh), self.urg_pointer)
 
        self.ip_header = final_ip_header
        self.tcp_header = final_tcp_header
        self.packet = final_ip_header + final_tcp_header
 
    def send_packet(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
        s.sendto(self.packet, (self.dest_ip, 0))
        data = s.recv(1024)
        s.close()
        return data
 
 
# could work with e.g. struct.unpack() here
# however, lazy PoC (012 = [SYN ACK]), therefore:
def check_if_open(port, response):
    cont = binascii.hexlify(response)
    if cont[65:68] == b"012":
        print("Port " + str(port) + " is: open")
    else:
        print("Port " + str(port) + " is: closed")
 
 
for port in [21, 22, 80, 8080]:
    # p = Packet("127.0.0.1", "172.17.171.8", port)
    p = Packet("172.17.171.8", "172.17.174.30", port)
    p.generate_packet()
    result = p.send_packet()
    check_if_open(port, result)
```

> Port 21 is: closed
> Port 22 is: closed
> Port 80 is: open
> Port 8080 is: closed

**发包时必须指定本机ip地址，而不能用127.0.0.1。**

tcpdump抓包：https://blog.csdn.net/qq_36119192/article/details/84996511

### 总结

在Windows下发不了Raw Socket，不知道什么傻原因，协议族什么的设置不对。Linux下发包很快，但最后发现结果不准确，有些响应是端口开放的ip实际上并没有开放！所以，失败！

