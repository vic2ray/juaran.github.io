---
title: {{ title }}
date: 2022-04-08 14:09:37
tags:
---

## Mozilla

[https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Methods/CONNECT](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Methods/CONNECT)

> 在 HTTP 协议中，**`CONNECT`** 方法可以开启一个客户端与所请求资源之间的双向沟通的通道。它可以用来创建隧道（tunnel）。
> 例如，**`CONNECT`** 可以用来访问采用了 [SSL (en-US)](https://developer.mozilla.org/en-US/docs/Glossary/SSL)([HTTPS](https://developer.mozilla.org/zh-CN/docs/Glossary/https))  协议的站点。客户端要求代理服务器将 TCP 连接作为通往目的主机隧道。之后该服务器会代替客户端与目的主机建立连接。连接建立好之后，代理服务器会面向客户端发送或接收 TCP 消息流。

## RFC7231

[https://datatracker.ietf.org/doc/html/rfc7231#section-4.3.6](https://datatracker.ietf.org/doc/html/rfc7231#section-4.3.6)

> CONNECT方法要求接收方建立一个隧道到由请求-目标确定的目的源服务器，如果成功，此后限制其行为，在两个方向上盲目转发数据包，直到隧道关闭。 隧道通常用于创建一个端到端的虚拟连接，通过一个或多个代理，然后可以使用TLS（传输层安全）来确保安全。
> CONNECT只适用于向代理发出的请求。 一个源服务器可能会回应一个 2xx (Successful)状态代码来表示连接已经建立。然而，大多数源服务器并没有实现CONNECT。

> 发送CONNECT请求的客户端必须发送严格的 请求-目标（[RFC7230]第5.3节）；也就是说，请求-目标 即请求目标只包括隧道目的地的主机名和端口号，用冒号隔开。 比如说

```bash
CONNECT server.example.com:80 HTTP/1.1
Host: server.example.com:80
```

代理服务器可以通过以下两种方式建立隧道：直接连接到请求目标，或者，如果配置为使用另一个

代理，通过将连接请求转发到下一个入站代理。

任何2xx（成功）响应都表明发送方（以及所有入站代理）**将在结束成功响应的标题部分的空白行之后立即切换到隧道模式；在该空行之后接收的数据来自请求目标的服务器。**除了成功响应之外的任何响应都表明隧道尚未形成，并且连接仍由HTTP控制。

当隧道中介检测到任何一方已关闭其连接时，隧道将关闭：代理必须尝试将来自关闭方的任何未完成数据发送到另一方，关闭两个连接，然后丢弃任何未交付的剩余数据。

可以使用代理身份验证来建立建造一条隧道。例如：

```bash
CONNECT server.example.com:80 HTTP/1.1
Host: server.example.com:80
Proxy-Authorization: basic aGVsbG86d29ybGQ=
```

建立到任意服务器的隧道有很大的风险，尤其是当目标是一个众所周知的或保留的TCP端口，而该端口不用于Web流量时。例如，连接到请求目标“[example.com:25](http://example.com:25/)”会建议代理连接到SMTP通信的保留端口；如果允许的话，这可能会欺骗代理转发垃圾邮件。支持CONNECT的代理应将其使用限制在有限的已知端口集或安全请求目标的可配置白名单内。

服务器不得在2xx（成功）响应中发送任何传输编码(Transfer-Encoding)或内容长度(Content-Length)字段以连接。客户端必须忽略成功响应中接收到的任何内容长度或传输编码头字段才能连接。

CONNECT请求消息中的有效负载没有定义的语义；在CONNECT请求上发送有效负载正文可能会导致一些现有实现拒绝该请求。

对CONNECT方法的响应不可缓存。

## Wikipedia

[https://zh.wikipedia.org/wiki/HTTP隧道](https://zh.wikipedia.org/wiki/HTTP%E9%9A%A7%E9%81%93)

**HTTP隧道**用于在被限制的网络连接（包括[防火墙](https://zh.wikipedia.org/wiki/%E9%98%B2%E7%81%AB%E5%A2%99)、[NAT](https://zh.wikipedia.org/wiki/%E7%BD%91%E7%BB%9C%E5%9C%B0%E5%9D%80%E8%BD%AC%E6%8D%A2)和[ACL](https://zh.wikipedia.org/wiki/%E5%AD%98%E5%8F%96%E6%8E%A7%E5%88%B6%E4%B8%B2%E5%88%97)）以及其他限制的情况下在两台计算机之间建立网络链接。该隧道通常由位于[DMZ](https://zh.wikipedia.org/wiki/DMZ)中的[代理服务器](https://zh.wikipedia.org/wiki/%E4%BB%A3%E7%90%86%E6%9C%8D%E5%8A%A1%E5%99%A8)中介创建。

隧道还可以使用受限网络通常不支持的[协议](https://zh.wikipedia.org/wiki/%E7%BD%91%E7%BB%9C%E4%BC%A0%E8%BE%93%E5%8D%8F%E8%AE%AE)进行通信。

HTTP隧道的常见形式是标准[HTTP CONNECT](https://zh.wikipedia.org/wiki/%E8%B6%85%E6%96%87%E6%9C%AC%E4%BC%A0%E8%BE%93%E5%8D%8F%E8%AE%AE)方式。在这种机制下，客户端要求HTTP代理服务器将[TCP](https://zh.wikipedia.org/wiki/%E4%BC%A0%E8%BE%93%E6%8E%A7%E5%88%B6%E5%8D%8F%E8%AE%AE)连接转发到所需的目的地。然后服务器继续代表客户端进行连接。服务器建立连接后，代理服务器将继续代理与客户端之间的TCP流。**只有初始连接请求是HTTP，之后服务器将仅代理建立的TCP连接。**

正是这种机制让使用HTTP代理的客户端可以访问[TLS](https://zh.wikipedia.org/wiki/TLS)网站（即[HTTPS](https://zh.wikipedia.org/wiki/HTTPS)）。

## 总结

* 发送这种被封装的数据（HTTP over TLS）一般都会使用HTTP隧道技术，所以要发请求先建隧道，而CONNECT就是是用来建立隧道的
* CONNECT是发给代理的而不是发给你想要访问的服务器的，CONNECT请求告诉代理你要和哪个服务建立连接，代理给你返回200以后，你后续发送的所有流量才会被代理转发给你本来想要访问的服务器。

[在HTTP协议中，为什么部分网站使用connect，而不是直接get？](https://www.zhihu.com/question/427710961/answer/1552540073)

[Evolution of HTTP — HTTP/0.9, HTTP/1.0, HTTP/1.1, Keep-Alive, Upgrade, and HTTPS](https://medium.com/platform-engineer/evolution-of-http-69cfe6531ba0)

# 迷惑

为什么Chrome浏览器打开network查看数据包看不到HTTPS站点的CONNECT请求？反之，为什么使用抓包工具如Charles、Fiddler等**代理抓包**软件能够看到HTTP-CONNECT连接？为什么抓包软件需要安装并信任本地证书后才能解析出HTTPS数据包？

现在假设浏览器请求情景，对于一个浏览器从未打开过的HTTPS站点，没有该站点的CA证书记录，当访问该站点时，将发生以下情景：

1. 首先会是域名解析，拿到站点的IP地址。发出UDP-DNS查询报文，查询结果被浏览器缓存
2. 浏览器向目标IP地址发出TLS-Client hello包，其中包含了SNI域名信息
3. 服务器回应TLS-Server hello，接下来是certificate，server key exchange, hello done
4. 浏览器回应TLS-Client key exchange, change cipher spec，encrypted message
5. 服务器回应TLS-Server change cipher spec, encrypted message，握手完成
6. 浏览器发送加密的HTTP-GET数据包，服务器发送加密的HTTP-Response
7. 浏览器收到HTTP-Response，解密明文HTTP，渲染页面

那么，最终在浏览器network看到的只有HTTP-GET和Response的明文信息了

什么？那HTTP-CONNECT在哪里？是的，浏览器拥有服务器证书，握手完成后直接发送了GET请求，并不需要发送CONNECT请求！因为**CONNECT只是发给代理服务器用于发起HTTPS请求的！**

---

最容易“搭建”的HTTP代理服务器就是抓包软件了。当浏览器设置了HTTP代理，监听本地的代理端口，浏览器所有的HTTP请求将流入到代理软件中，由代理软件完成真实请求并返回响应。过程如下：

1. 浏览器发起普通HTTP-GET请求，发现设置了代理
2. 浏览器将请求修改为符合代理协议的请求。浏览器发给代理服务器的请求是这样的：

```bash
GET <http://www.example.com/index.html> HTTP/1.1
User-Agent: xxxx
Host: www.example.com
Proxy-Connection: Keep-Alive
```

代理请求的URL 必须是 **完整路径** ，这个也是 Http1.0 规范。至于Proxy-Connection设置长连接：

> http1.0 时代的产物。老旧的代理，如果设置 connection: keepalive，代理原样转发给服务器，服务器会以为要建立长久连接，但是代理并不支持，这样就出问题了。
> 所以改为设置 proxy-connection: keepalive，如果是新的代理，支持 keepalive，它会认得这个头，并改成 connection: keepalive 转发给服务器，顺利建立持久连接；如果是老的代理，它不认识，会原样转发，这时候服务器也不会建立持久连接。完美。

参考：****[浏览器什么时候会在 http 请求头中添加 proxy-connection: keepalive](https://www.v2ex.com/t/411098)****

1. 代理服务器收到这个GET请求后，重新组装这个数据包为标准的HTTP-GET并发送：

```bash
GET /index.html HTTP/1.1
User-Agent: xxxx
Host: www.example.com
Connection: Keep-Alive
```

1. 目标服务器收到请求后响应给代理服务器`HTTP/1.1 200 OK`以及内容，原路返回给本地浏览器。

而至于Client-Proxy-Server之间的TCP长连接如何处理，协议标准是这么说的：

> The proxy server MUST signal persistent connections **separately** with its clients and the origin servers (or other proxy servers) that it connects to. Each persistent connection applies to only one transport link.

代理服务器分别处理与其连接的客户端和源服务器的TCP持久连接。

---

那么对于HTTPS请求呢？HTTPS是加密HTTP后的TCP数据，在没有证书的情况下，代理将不能直接解析出里面的HTTP明文，代理过程如下：

1. 浏览器发现请求的是https站点，那么就必须先发出TLS-Client hello
2. 在发出TLS握手包前，发现设置了HTTP代理服务器
3. 浏览器将首先与代理服务器建立TCP三次握手连接
4. 浏览器发出HTTP-CONNET请求，这是个普通的明文HTTP，如下：

```bash
CONNECT server.example.com:80 HTTP/1.1
Host: server.example.com:80
```

是的，就只包含了CONNECT头和Host字段，表明了浏览器将与那个主机发起TLS连接

1. 代理服务器收到CONNECT后，新建与目标主机的TCP握手连接
2. 因为代理服务器既和浏览器建立了TCP连接，又和服务器建立了TCP连接，这就是一条TCP-Tunnel隧道，将用来传输TCP数据，而代理服务器只作转发
3. 浏览器发出TLS-Client hello，经由隧道流至源服务器
4. 源服务器响应TLS-Server hello，完成TLS握手过程
5. 浏览器最终发出加密的HTTP-GET

在这个过程中， 代理服务器收到CONNECT请求后建立TCP tunnel，之后的所有流量都是TCP，因此代理服务器并不能识别出任何有效的通信数据（最多只能够识别出SNI）。

---

而如果在本地安装信赖了代理服务器的证书，情况就大不一样了。

1. 浏览器本应该发出CONNECT要求代理服务创建TCP隧道
2. 但代理服务器说，跟我建立TLS就行！我把我的证书给你，你信任一下
3. 于是浏览器跟代理建立了TLS连接，并发出加密数据
4. 代理收到后解出了要访问的内容，于是又以客户端的身份向源站发起TLS连接，
5. 代理使用源服务器的CA解密并得到明文结果
6. 代理再把结果加密后返回给浏览器，浏览器用代理的CA解密出明文响应

---

简单总结一下:

* 对于普通的HTTP站点，代理接受完整URL请求首行的代理HTTP报文格式、编辑为标准HTTP报文重新发送。使用这种方式完成的请求被称为 **HTTP代理方式** 。
* CONNECT是用于告知代理服务器建立客户端与服务端的TLS通信的一种HTTP请求方式。使用这种方式完成客户端与源服务器通信的方式被称为 **HTTPS代理** 。

那说了半天，安装证书的代理方式呢？代理服务器本身拥有证书，客户端信任该证书，从而客户端与代理建立TLS连接通信，代理与源服务器建立TLS连接通信，而不再是代理建立TCP隧道，也就不需要CONNECT了。这种方式的本质上属于 **中间人攻击** ，而不是可靠的代理了。因为需要客户端安装并信任中间人的自签证书，所以一般用于本地HTTPS流量抓包。

最后，在代码中设置代理请求时，一般会设置这两种代理。下面设置的意思是，当访问站点属于http站点时，使用http代理；当访问站点为https时，使用https代理方式。如果只设置了其中一个，例如只设置http，那么当遇到https时就不会使用这个代理来请求，而是本机直接连接。

```python
proxies = {
		"http": "cloundnproxy.baidu.com:443",
		"https": "cloundnproxy.baidu.com:443"
}
requests.get(url, proxies=proxies)
```

[http 代理和 https 代理的区别](https://www.v2ex.com/t/593765)

# 代理图示

* HTTP代理

![](https://cdn.jsdelivr.net/gh/Juaran/juaran.github.io@image/notion/20220408141204.png)

* HTTPS代理
  ![](https://cdn.jsdelivr.net/gh/Juaran/juaran.github.io@image/notion/20220408141426.png)

**[HTTP、HTTPS代理分析及原理](https://lilywei739.github.io/2017/01/25/principle_for_http_https.html)
[HTTP代理原理分析](https://www.cnblogs.com/xugongzi007/p/12802819.html)**

