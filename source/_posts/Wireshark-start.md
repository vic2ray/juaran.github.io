---
title: 从Wireshark开始
date: 2021-09-14 21:10:00
category: 计算机网络
tag: proxy
---
通过”观察协议的运作“和“玩转协议”可以大大加深我们对网络协议（network protocol）的理解——观察两个协议体（protocol entity）之间消息序列的交换、深入研究协议操作的细节、控制协议完成特定的行为并观察这些行为产生的后果。这个过程可以在沙箱环境中进行，也可以在正式网络环境中完成。通过Wireshark近距离地观察协议，与互联网中其他协议实体交互和交换信息。

Hackers经常使用**包嗅探**工具（packet sniffer）窥视网络中发送的消息。Wireshark将获取从你电脑发出或接收到的每一个链路层帧（link-layer frame）的**拷贝**。由HTTP、FTP、TCP、UDP、DNS、或IP交换的消息最终都封装在链接层帧中，这些帧通过以太网电缆等物理媒体传输。

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210914200427447.png" alt="image-20210914200427447" style="zoom: 80%;" />

因为计算机已经实现了所有不同层的协议，所以您将可以访问链路层包，例如一个以太网包。Wireshark可以嗅探到这些包，然后从中提取IP数据报（IP datagram），提取传输层段（transport layer segment），如TCP段等。

观察正在执行的协议实体之间交换的消息的基本工具称为数据包嗅探器。顾名思义，包嗅探器会捕获从计算机发送/接收到的（“嗅探”）消息；它通常还会存储或在这些捕获的消息中显示各种协议字段的内容。数据包嗅探器本身是**被动**（passive）的。**它只观察在计算机上运行的应用程序和某种协议正在发送和接收的消息，但不发送数据包本身。类似地，接收到的数据包从未显式地发送到数据包嗅探器。相反，数据包嗅探器接收的是在机器上执行的应用程序的协议发送/接收的数据包的副本。**

下图是包嗅探器的工作模式。虚线上方是计算机的应用层，常用的应用如web浏览器、ftp客户端；虚线下方是操作系统层，pcap（package capture）是用于捕获计算机链路层数据帧的操作系统级API。假设物理传输介质是以太网，所有的上层协议最终都被封装在以太网框架中。

> Windows系统：WinPcap；Unix系统：libpcap

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210914201510795.png" alt="image-20210914201510795" style="zoom:80%;" />

包嗅探器的第二个部分是数据包分析器（packet analyzer），用于显示协议消息中的所有字段的内容。为了显示协议字段内容，分析器必须能够”理解“所有通过协议传输的消息的结构。例如，要读懂一个HTTP协议报文消息，分析器必须了解以太网帧（Ethernet frames）的协议格式，并识别以太网帧内的IP数据包（IP datagram）的格式，从中提取出TCP段（segment），最后根据TCP协议结构提取出HTTP消息，理解HTTP协议，开头字节将是“GET”、“POST”等请求方法。

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210914204934434.png" alt="image-20210914204934434" style="zoom:80%;" />