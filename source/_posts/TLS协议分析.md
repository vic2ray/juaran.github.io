---

title: TLS协议分析
date: 2022-4-4
category: 计算机网络

---



# TLS

TLS协议包括两个协议组：TLS记录协议(TLS Record)和TLS握手协议(TLS Handshake)

## Client Hello

客户端发送hello包的主要信息：

- 协议版本
- 客户端随机数据（稍后在握手中使用）
- 要恢复的可选会话 ID
- 密码套件列表
- 压缩方法列表
- 扩展列表

<!--more-->

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled.png" style="zoom:67%;" />

- Content Type：Handshake(0x16)表示这是一个握手协议包
- Version：TLS协议版本号，通常为TLS 1.0而非预期的TLS 1.2。golang crypto/tls库中提到：一些TLS服务器在握手阶段失败如果高于TLS 1.0
- Length：TLS协议包长度
- 接下来是Handshake Protocol协议内容
    - Handshake Type：0x01代表Client Hello
    - Length：握手包长度
    - Version：客户端TLS版本号。SSL 3.3=TLS 1.2，SSL 3.1=TLS 1.0，SSL 3.0之后开始规范化
    - Random：客户端提供 32 字节的随机数据（时间戳+随机数），将作为服务端生成交换密钥的因子
    - Session ID：用来表明一次会话，第一次建立没有。如果以前建立过，可以直接带过去以复用连接资源
    - Cipher Suites：客户端支持的加密-签名算法的列表，让服务器去选择。例如：
        - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
            - ECDHE_RSA：非对称加密算法，密钥协商，用于密钥交换
            - AES_128_GCM：握手完成后的数据加密算法
            - SHA256：数字证书校验的数字签名哈希算法
    - Compression Method：此功能已从未来的 TLS 协议中删除
    - Extensions Length：扩展信息长度。客户端提供了一个可选扩展列表，服务器可以使用这些扩展来采取行动或启用新功能。
    - Extension：扩展列表
        - **server_name**：客户端提供了它正在连接的服务器的域名，也称为 **SNI**（服务器名称指示，Server name indication）。如果没有此扩展，HTTPS 服务器将无法为单个 IP 地址（虚拟主机）上的多个主机名提供服务，因为在协商 TLS 会话并发出 HTTP 请求之前，它无法知道要发送哪个主机名的证书。
            - 00 - 列表条目类型为 0x00“DNS 主机名”
            - 00 13- 0x13 (19) 个字节的主机名如下
            - 65 78 61 ... 6e 65 74 - “example.ulfheim.net”
        - status_request：客户端允许服务器在其响应中提供 OCSP 信息。OCSP 可用于检查证书是否已被吊销
        - elliptic_curves：客户表示它支持 4 条曲线的椭圆曲线 (EC) 加密
        - ec_point_formats：椭圆曲线加密细节协商
        - signature_algorithms：此扩展指示客户端能够理解哪些签名算法，并可能影响服务器发送给客户端的证书的选择

## Server Hello

服务器回复“你好”。服务器提供以下内容：

- 选择的协议版本
- 服务器随机数据（稍后在握手中使用）
- 会话 ID
- 选定的密码套件
- 选择的压缩方法
- 扩展列表

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 1.png" style="zoom:67%;" />

- 记录头信息：这是一个TLS握手记录、TLS版本、握手记录长度
- 握手类型：0x02（服务器问候）；握手信息长度；版本信息
- Random：与Client Hello随机数功能一样，将用于对方的交换密钥生成
- Session ID：服务器可以为此会话提供一个 ID，客户端可以在以后的会话协商中提供该 ID，以尝试重新使用密钥数据并跳过大部分 TLS 协商过程。为此，服务器和客户端都将来自先前连接的关键信息存储在内存中。恢复连接可以节省大量计算和网络往返时间，因此尽可能执行
- Cipher suite：服务器从客户端提供所支持的密码套件中选择一个
- 扩展信息

## Certificate,Server Key Exchange,Server Hello Done

这次传输同时包含了三部份TSL Records：

- Certificate
- Server Key Exchange
- ServerHello Done

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 2.png" style="zoom:67%;" />

### Certificate

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 3.png" style="zoom:67%;" />

- 这是一个TLS握手包且握手类型为0x0B（证书）
- Certificates：由一组证书组成的证书链，最上面的是服务端本身的证书，依次往下是：中级证书机构证书、根证书机构证书。客户端将提取出证书中的公钥、数字签名信息
- Certificate：证书格式必须是X.509，格式如下：
    - version：证书版本号
    - serialNumber：证书序列号，这个每个颁发机构是唯一的
    - signature：签名算法，sha256WithRSAEncryption，即对证书进行Sha256哈希生成消息摘要，然后使用CA机构的RSA私钥生成数字签名。客户端将使用该哈希算法计算消息摘要
    - issuer name：证书办法机构信息
    - validity：证书有效期
    - subject：证书持有者信息，即服务器提供者，包括域名、地区等信息
    - subjectPublicKeyInfo：CA证书公钥算法及公钥值，用于客户端校验消息摘要
    - algorithmIdentifier：证书签名算法即数字签名值

证书信息在浏览器上能够看的更详细，和上面一致

### Server Key Exchange

到这里，客户端已经告知了服务器能够支持的加密套件，且服务器也发送给了客户端证书。客户端通过正向Sha256计算证书的消息摘要，和反向CA公钥计算证书签名的消息摘要，发现该证书值得信赖。那么下一步就是放心的交换非对称加密公钥的时候了，也称密钥交换。

作为密钥交换过程的一部分，服务器和客户端都将拥有一对公钥和私钥，并将向对方发送他们的公钥。然后将使用每一方的私钥和另一方的公钥的组合生成共享加密密钥。

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 4.png" style="zoom:67%;" />

- 这是一个TLS握手包，握手信息为服务器的密钥交换(0x0C)
- Hello阶段已经协商使用ECDHE椭圆曲线非对称加密算法
    - Curve Type、Named Curve：指定椭圆曲线类型
    - Pubkey：公钥信息，即交换到给客户端的公钥
    - Signature：为了证明服务器拥有服务器证书（在此 TLS 会话中提供证书有效性），它使用证书的私钥签署临时公钥。客户端可以使用证书的公钥验证此签名。

### Server Hello Done

- 这是一个TLS握手包，握手信息为服务器已经Hello完成(0x0D)

## Client Key Exchange,Change Cipher Spec,Encrypted Handeshake Message

第一阶段：客户端服务器互发Hello，确定加密套件

第二阶段：服务端发来证书和公钥，客户端回应公钥，完成密钥交换

在客户端回应的TLS包中，同样保含了三个TLS Layers：

### Client Key Exchange

在上一步服务器已经发来了ECDHE公钥即验证签名，客户端利用证书公钥验证发来的交换公钥是否真实。客户端生成自己的ECDHE公钥和私钥，并发送公钥给服务端：

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 5.png" style="zoom:67%;" />

- 这是一个TLS握手报文，类型为客户端密钥交换(0x10)
- 客户端公钥信息。这里不需要向服务端提供验证签名

在发送完公钥后，客户端将利用已有信息计算后续数据通信的对称加密密钥。客户端现在拥有计算每一方将使用的加密密钥的信息。它在此计算中使用以下信息：

- 服务器随机（来自 Server Hello）
- 客户端随机（来自Client Hello）
- 服务器公钥（来自服务器密钥交换）
- 客户端私钥（来自客户端密钥生成）

计算细节不展示。计算的结果称为master_secret，而且在服务端收到公钥后，同样需要计算得到相同的master_secret。这是密钥协商、交换后得到的保密密钥。

### Change Cipher Spec

客户端表明它已经计算了共享的加密密钥，并且来自客户端的所有后续消息都将使用客户端写入密钥进行加密。

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 6.png" style="zoom:67%;" />

- 这只是间通知事件，告诉服务端，我这边已经计算好对称加密密钥了，你那边也计算好，之后我们的加密就用master_secret了！

### Encrypted Handshake Message

这一步对应的是 Client Finish 消息，客户端将前面的握手消息生成摘要再用协商好的秘钥加密，这是客户端发出的第一条加密消息。服务端接收后会用秘钥解密，能解出来说明前面协商出来的秘钥是一致的。

验证数据是根据所有握手消息的哈希构建的，并验证握手过程的完整性。

## Server Change Cipher Spec,Encrypted Handshake Message

- 服务器回复说，我知道了，我已经计算好了master_secret；
- 我已经解密了你的数据，我现在也发一条加密好的给你

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 7.png" style="zoom:67%;" />

至此，TLS-Handshake阶段正式完成！

## Application Data

接下来就真正的到了接口请求的阶段。TLS的Content-Type为Application Data。 传输的内容也是加密的：

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 8.png" style="zoom:67%;" />

# TLS图示

<img src="https://cdn.jsdelivr.net/gh/juaran/juaran.github.io@image/notion/Untitled 9.png" style="zoom: 50%;" />

Wireshark抓包未能成功获取完整的握手建立过程，因为在浏览器访问目标站点时已经有了证书和Session ID缓存。在下一次连接时，客户端只需要发送Client Hello包，并携带Session ID和Session Tickets信息即可访问服务器

[The Transport Layer Security (TLS) ProtocolVersion 1.2](https://www.rfc-editor.org/rfc/rfc5246.html)

[今天我抓了个 HTTPS 的包](https://www.cnblogs.com/flashsun/p/15347982.html)

[Https详解+wireshark抓包演示](https://www.jianshu.com/p/a3a25c6627ee)

[开启 TLS 1.3 加密协议，极速 HTTPS 体验](https://zhuanlan.zhihu.com/p/32987630)