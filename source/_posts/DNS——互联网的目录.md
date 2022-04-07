
---
title: DNS——互联网的目录
date: 2021-10-04
category: 计算机网络
---

> 内容摘自：*Computer Networking: A Top-Down Approach, 6th* *ed.,* J.F. Kurose and K.W. Ross
>
> Chapter **2.5** **DNS—The Internet’s Directory Service**

<!-- more -->

网络中计算机的身份证：

* hostname：不定长度，包含有限的可能有意义的身份信息
* IP address：固定长度，通过路由器传递下一跳(next hoop)信息

### Services Provided by DNS

我们需要这样一个可以将hostname转换为IP地址的**目录服务**，它叫域名系统(Domain Name System)，它是：

* 具有分层结构的分布式数据库
* 用于主机查询分布式数据库的应用层协议

**DNS server**通常是运行BIND(Berkeley Internet Name Domain)软件的计算机，BIND 9下载地址：https://www.isc.org/download/#BIND。DNS协议基于UDP协议，默认监听53端口。DNS协议通常在其他协议中使用，包括：HTTP、SMTP、FTP等，用于转换hostname为IP address，以便于创建TCP连接——这意味着DNS带来了额外的查询延时。

除了域名解析，DNS还用于其他应用场景：

* 主机别名：一个固定IP address的主机可以解析到不同的别名
* 邮件服务器别名：163邮箱的真实域名为：163mx02.mxmail.netease.com，但解析到163.com更便捷
* **负载分发**：同一个域名关联到一组IP地址，使用多个Web服务器均衡负载；或使用在CDN中分发内容。

### Overview of How DNS Works

从总体来看，DNS为其他应用层服务如HTTP提供域名-IP地址解析服务，或通过`gethostbyname()`函数调用，本地DNS客户端收到解析请求后，发送一条UDP数据报查询DNS数据库得到响应报文，然后转发结果给调用方。

一个简单的设计是，将所有的hostname-IP address映射放在一台中心DNS服务器中，这显然是最糟糕的设计：

* 单点服务器的最大弱点是，一旦宕机全部玩完
* 流量限制、距离限制、难以维护和扩展

#### **A Distributed, Hierarchical Database**

实际中有大量的DNS服务器，通过分层的方式组织起来，并分布在世界各地。分层形式如图所示：

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211004152815418.png" alt="image-20211004152815418" style="zoom:80%;" />

* Root DNS servers: 分布在全球各地的“13”个根DNS服务器，用于存储TLD服务器的映射信息。实际上出于安全性考虑，每个根域名服务器都多个备份服务器。其中大部分在北美

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211004155244910.png" alt="image-20211004155244910" style="zoom: 80%;" />

* Top-level domain (TLD) servers: 顶级域名服务器：com,org,net,edu,gov,us,fr,cn,jp...由非盈利组织监管
* Authoritative DNS servers：权威DNS服务器。在互联网中具有公开访问主机的组织机构都必须提供公开的DNS记录（否则用户无法访问），机构DNS服务器中包含了主机的映射。大多数大学和大公司实现并维护他们自己的主和次（备份）权威DNS服务器

尽管严格意义上来说，**Local DNS server**并不属于以上DNS的分层结构中，但本地DNS服务器依然是DNS架构中的核心部分。本地DNS服务器往往是ISP提供的，例如手机网络运营商有自己的域名服务系统，校园网教育网有自己的DNS服务器，阿里巴巴等大公司也有自己的DNS服务器。在电脑的网络适配器中可以指定DNS服务器的IP地址，这些DNS服务器往往距离主机最近，以获得最快的查询响应。Local DNS server更主要的作用是在主机进行DNS查询时，**充当代理**将查询转发到DNS层级结构中。

#### Recursive queries and iterative queries

在实际中，即使用递归也使用迭代查询：从请求主机到本地DNS服务器的查询是递归的，其余的查询是迭代的（左图模式）。这样本地DNS服务器承担了局部网络的大部分的查询压力，减轻了层级域名服务器的压力。

<table><tr><td>
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211004161718223.png" style="zoom: 67%" />
</td><td>
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211004161749555.png" -image/raw/master/typora/image-20211004161718223.png" style="zoom: 67%" />
</td></tr></table>

#### DNS Caching

上一章我们深入理解了Web caching机制，同样的，DNS缓存的目的是为了降低请求延迟和减少整个网络中重复的DNS查询。通常在Local DNS server中启用缓存，并设置缓存更新频率。

### DNS Records and Messages

#### DNS records

DNS服务器中存储的是资源记录（RRs, resource records），也就是主机到IP地址的映射记录，DNS响应消息中携带了一条或多条这样的记录。每一条资源记录都是四元组组成：`(Name, Value, Type, TTL)`. 其中TTL(time to live)是记录的存活时间，到0时会从缓存中删除。我们重点关注Name, Value和Type：

* Type=A：标准的hostname-to-IP address映射
* Type=NS：提供domain-to-authorized DNS server映射，例：(fzu.edu.cn, func.fzu.edu.cn, NS)
* Type=CNAME：Canonical 规范主机名的Alias 别名解析
* Type=MX：邮件服务器的别名解析，例：(163.com, 163mx00.mxmail.netease.com, MX)

Unix主机使用`nslookup`命令可以进入查询交互。

#### **DNS Messages**

DNS查询和响应的报文格式是一样的：

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211004170729358.png" alt="image-20211004170729358" style="zoom: 67%;" />

Wireshark抓包：

<table><tr><td>
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211004172204416.png" alt="image-20211004172204416" style="zoom:67%;" />
</td><td>
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20211004172426338.png" alt="image-20211004172426338" style="zoom:67%;" />
</td></tr></table>
### Summary

DNS是计算机网络中最重要的基础服务设施之一，多年来针对DNS服务器的DDoS、中间人攻击等的成功防御都证明了DNS在面临攻击时的强大抵抗力，这得益于其三层分级结构和分布式部署，以及本地DNS服务器的缓冲保护。



参考文章：[DNS(bind)服务器的安装与配置](https://blog.csdn.net/bbwangj/article/details/82079405)

