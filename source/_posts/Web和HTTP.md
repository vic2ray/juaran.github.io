
---
title: Web和HTTP
date: 2021-09-25
category: 计算机网络
---


> 内容摘自：*Computer Networking: A Top-Down Approach, 6th* *ed.,* J.F. Kurose and K.W. Ross
>
> Chapter 2.2 The Web and HTTP

<!-- more -->
 
在计算机网络的浩瀚海洋中，DNS服务提供了计算机之间快速访达的目录。而我们大多数时候不直接通过DNS进行交流，而是通过应用程序如：Web、文件传输、电子邮件间接地调用DNS。DNS很好的诠释了如何在网络应用层上实现核心的网络功能（网络地址到网络域名的转换）。
 
计算机网络即使发展到20世纪90年代初，其使用者也仅仅局限于研究人员、学者和大学生用来登录到远程主机，进行文件传输、接收和发送电子邮件。1994年，Berners Lee为世界带来了the World Wild Web（WWW，简称Web），它极大地改变了人们与计算机的交互方式，将互联网的众多数据网络进化到唯一的数据网络。
 
其中最大的改变莫过于，人们从传统的广播电视、收音机被动接收信息的方式转为自发地、按需地通过Web获取信息，人们可以通过一个Hyper Link去往世界上的任意一个角落，和任何一个人交谈。图像、音频、视频，HTML、CSS、JavaScript，时刻刺激着人们的感官。
 
### HTTP
 
HyperText Transfer Protocol是Web的应用层协议，也是整个Web的核心。HTTP定义了Web客户端与Web服务端交换信息的方式和消息的结构。在Web中，最终展现给用户的是Web page，或称为document，一个网页文档可以是一个HTML文件、JPEG图片等；URL指定了每一个文档对象的路径；一个Web browser是实现了HTTP的Web客户端，不同的浏览器对于HTTP标准的实现细节有所不同，对HTML、CSS、JavaScript等标准协议的实现也有差异；Web server是HTTP的服务端实现，诸如Apache、IIS等。
 
HTTP使用TCP作为其底层传输协议(而不是运行在UDP之上)。**HTTP客户端首先启动与服务器的TCP连接**。一旦建立，浏览器和服务器**通过其socket接口处理访问TCP**。Socket是沟通上层HTTP与下层TCP的门，浏览器将HTTP发送给socket接口，并从socket接口中获取HTTP响应消息，服务端亦然。一旦HTTP从客户端流向socket接口，TCP即接管了数据报，因为TCP的可靠传输服务，浏览器不必担心数据丢失、网络重连等问题，这就是协议分层架构带来的巨大好处。
 
因为连接信息由TCP接管，HTTP请求发出去后没有存储任何记忆信息，服务器作出响应后也没有存储状态信息，因此HTTP是**stateless protocol**, 只要发出了请求，服务器就无条件作出响应。
 
### Non-Persistent Connections
 
在Client-Server模型中，双方通信的持久性是需要考虑的问题。连续、定期或间歇性地发出请求/响应？每个请求/响应通过单独的TCP连接发送（非持久连接），还是所有请求都通过一个TCP连接发送（持久连接）？HTTP既可以使用non-persistent连接，也可以使用persistent连接，默认情况下使用持久性连接。
 
在一个HTML文档中，有10张图片的引用链接，客户端每次通过socket接口建立与服务端socket接口的TCP连接，接收到文档后关闭TCP连接，一共经历了11次TCP连接和断开。这是非持久连接的情况，即每个TCP连接只处理一个HTTP请求和响应。一般情况下，浏览器会打开5-10个并行的TCP连接来进行请求-响应事务，但网络开销并未减少。我们使用RTT（round-trip time）往返延时来衡量一个数据包从发出到接收的时间花销，经过TCP三次握手和文档传输，一共消耗两个RTT+文件传输时间。
 
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210925111950922.png" alt="image-20210925111950922" style="zoom:80%;" />
 
如果使用持久连接，每次请求文档时不需要初始化连接，可以节约一个RTT交付延时和TCP缓冲区。持久连接为同一个Web页面上的请求使用同一TCP连接，此外，对于同一个服务器但不同网页的请求也将使用同一TCP连接。具体来说，浏览器会在HTTP的header lines中默认添加`Connection: Keep-Alive`使用长连接，并可以通过`Keep-Alive: timeout=600`来设置连接过期时间。
 
### Message Format
 
<table><tr><td>
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210925114206842.png" alt="image-20210925114206842" style="zoom: 100%;" />
</td><td>
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210925114224546.png" alt="image-20210925114224546" style="zoom:100%;" />
</td></tr></tabel>
 
### Cookies
 
我们提到HTTP是无状态stateless的协议，这简化了服务器设计，允许Web server能够高效处理数千个TCP连接，带来的问题是无法通过连接区分用户，因而使用cookies用来跟踪用户。
 
Cookies存在四个部分中：
 
1. 服务器生成用户的cookies信息，并保存到后端数据库中
2. 服务器响应时在header line中添加：`set-cookie: user:xxx;name:xxx `
3. 客户端（浏览器）保存cookies信息到访问站点的本地Cookies存储空间
4. 客户端请求时在header line中添加：`cookie: user: xxx;name:xxx`
 
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210925115132704.png" alt="image-20210925115132704" style="zoom: 67%;" />
 
由此，Cookie在无状态的HTTP之上建立了一个用户会话层。Cookie同时也带来了用户隐私问题，服务提供方可以通过cookie生成用户画像，提供给第三方。例如，在微信中出现了京东的广告，那么京东将用户cookie共享给了微信；访问百度贴吧时可能出现百度百科或其他百度系产品的信息，这些不同站点而却存储了其他站点的cookie信息，称为第三方cookie。
 
### Web Caching
 
Web cache，也称为Proxy server，是对源Web服务器进行代理的独立网络实体。一台代理服务器拥有自己的存储，用于保存源服务器上的内容，当收到客户端请求时返回本地存储的内容，否则请求源服务器并返回。Proxy server既是客户端，又是服务端。
 
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210925160213500.png" alt="image-20210925160213500" style="zoom:80%;" />
 
通常，Web缓存服务由ISP（互联网服务提供商）建立，大型网络机构也可以建立自己的Web缓存服务器。Web缓存可以大大减少客户端请求的响应时间，特别是当客户端和服务端的瓶颈带宽远小于客户端和Web缓存服务器间的瓶颈带宽时。其次，Web缓存大大降低了由局域网机器直接接入到互联网中的流量，从而减轻服务器负担，也降低了升级外网带宽的成本。
 
<table>
<tr><td>
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210925161635629.png" style="zoom:70%;" />
</td><td>
<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/typora/image-20210925161654525.png" style="zoom:70%;" />
</td></tr></table>
 
通过使用内容分发网络(CDN)，Web缓存在互联网中发挥着越来越重要的作用。一家CDN公司在整个互联网上安装了许多地理上分布的缓存，从而本地化了大部分流量。
 
进而思考，由于Web cache缓存的是源服务器上内容的副本，当源服务器上内容发生改变时，需要一种机制来验证客户端从Web cache获取到的内容是否为最新。这个机制就是Conditional GET（条件GET）。具体过程为：
 
1. 浏览器发起内容请求，通常为普通的HTTP GET请求
 
2. Web cache收到请求：
 
   * 若无内容拷贝，请求源服务器，响应header lines中包含`Last-Modified: `，Web cache保存对象的拷贝并记录最后修改日期，返回给浏览器
 
   * 若有内容副本，向源服务器发送一条header lines中包含`If-Modified:`的请求：
     * 若源服务器返回：`304 Not Modified`和空的响应对象，则直接返回浏览器上一次的副本
     * 若源服务器返回：最后修改日期和对象，则更新拷贝并返回浏览器
 