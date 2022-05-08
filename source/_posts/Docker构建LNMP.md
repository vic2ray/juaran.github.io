---
title: {{ title }}
date: 2022-05-08 12:24:19
tags: toy
---
使用Docker构建常见的Linux + Nginx + Mysql + PHP环境。以Metinfo企业产品展示站点为例

思路：

1. pull下载nginx、mysql5.7、php-fpm7.2镜像
2. 项目映射到nginx内/var/www/html
3. 配置nginx，将所有.php的请求交给php-fpm容器处理
4. php-fpm内安装mysqli扩展，修改连接信息主机名为mysql容器名
5. 使用docker桥接网络将三个容器置于同一网络中

手动构建以上过程，然后编写Dockerfile自动构建容器，最后使用docker-compose自动化部署

## Docker

首先下载主要镜像：

```bash
docker pull nginx
docker pull mysql:5.7
docker pull php:7.2-fpm
```

分别启动容器。先启动一个nginx容器试试：

```bash
docker run -d -p 80:80 nginx
```

**Welcome to nginx!** 然后我们进入容器内一探究竟。配置文件放在`/etc/nginx`目录下，读取的默认配置文件为`/etc/nginx/conf.d/default.conf` ，内容如下：

* default.conf
  ```bash
  server {
      listen       80;
      listen  [::]:80;
      server_name  localhost;
  
      #access_log  /var/log/nginx/host.access.log  main;
  
      location / {
          root   /usr/share/nginx/html;
          index  index.html index.htm;
      }
  
      #error_page  404              /404.html;
  
      # redirect server error pages to the static page /50x.html
      #
      error_page   500 502 503 504  /50x.html;
      location = /50x.html {
          root   /usr/share/nginx/html;
      }
  
      # proxy the PHP scripts to Apache listening on 127.0.0.1:80
      #
      #location ~ \\.php$ {
      #    proxy_pass   <http://127.0.0.1>;
      #}
  
      # pass the PHP scripts to FastCGI server listening on 127.0.0.1:9000
      #
      #location ~ \\.php$ {
      #    root           html;
      #    fastcgi_pass   127.0.0.1:9000;
      #    fastcgi_index  index.php;
      #    fastcgi_param  SCRIPT_FILENAME  /scripts$fastcgi_script_name;
      #    include        fastcgi_params;
      #}
  
      # deny access to .htaccess files, if Apache's document root
      # concurs with nginx's one
      #
      #location ~ /\\.ht {
      #    deny  all;
      #}
  }
  ```

其中重要信息`location /`指向了`/usr/share/nginx/html` ，也就是站点目录；注释部分说明了如果匹配到.php结尾的请求将转交给FastCGI处理，监听地址127.0.0.1:9000。那么我们取消这一段注释，然后开启php-fpm容器并监听9000端口：

```bash
docker run -d -p 9000:9000 --name=lnmp-phpfpm php:7.2-fpm
```

在nginx站点目录下新建一个index.php：

```php
<?php
    echo phpinfo();
?>
```

然后再访问站点，结果直接跳出下载inedx.php了，说明在nginx容器内并没有访问到php-fpm容器。因为nginx容器内的127.0.0.1:9000显然并不是容器外的9000端口的php-fpm服务。一种做法是获取php容器ip:172.16.16.3填入，但这样无法适应所有环境。我们创建一个docker桥接网络，名字就叫lnmp，然后把nginx和php容器加入该网络中：

```bash
docker network create lnmp
docker network connect lnmp lnmp-nginx
docker network connect lnmp lnmp-phpfpm
```

这样，在nginx容器内只需要替换127.0.0.1为lnmp-phpfpm，就能自动解析php容器的ip地址了。

另外一个坑是nginx调用php-fpm时默认配置的php文件路径为：

```bash
fastcgi_param  SCRIPT_FILENAME  /scripts$fastcgi_script_name;
```

网上教程说需要将`/scripts`替换为`$document_root` 。但是依然会报错：File not found.

**Nginx+FastCGI运行原理**

> Nginx不支持对外部程序的直接调用或者解析，所有的外部程序（包括PHP）必须通过FastCGI接口来调用。FastCGI接口在Linux下是socket，（这个socket可以是文件socket，也可以是ip socket）。为了调用CGI程序，还需要一个FastCGI的wrapper（wrapper可以理解为用于启动另一个程序的程序），这个wrapper绑定在某个固定socket上，如端口或者文件socket。当Nginx将CGI请求发送给这个socket的时候，通过FastCGI接口，wrapper接纳到请求，然后派生出一个新的线程，这个线程调用解释器或者外部程序处理脚本并读取返回数据；接着，wrapper再将返回的数据通过FastCGI接口，沿着固定的socket传递给Nginx；最后，Nginx将返回的数据发送给客户端，这就是Nginx+FastCGI的整个运作过程。

因此，fastcgi_param传递的参数SCRIPT_FILENAME是执行php文件的路径而不是php文件。我们跨容器进行通信，则需要保证php-fpm能够接收到正确的路径。因此应该替换`/scripts`为php容器内默认路径`/var/www/html/` ，并且把整个项目文件夹映射到php容器内（实际工作时只会处理php）。

参考：**[Docker 安装 PHP 并配合 Nginx 运行 phpinfo](https://www.jiloc.com/45874.html)**

接下来新建mysql容器并让php连接mysql容器。创建并加入桥接网络：

```bash
docker run -d -e MYSQL_ROOT_PASSWORD="root" -p 3306:3306 --name=lnmp-mysql mysql:5.7
docker network connect lnmp lnmp-mysql
```

php安装mysql扩展插件，重启生效:

```bash
docker exec -it lnmp-phpfpm bash
docker-php-ext-install mysqli
docker restart lnmp-phpfpm
```

测试连接到mysql容器：~php代码中实际上读取的是主机名，并不是虚拟主机名，访问3306端口不走docker-network中的lnmp-mysql~ 在网页初始化部署时需要填数据库主机名，填入容器名lnmp-mysql即映射道容器ip

```php
<?php
	$link = mysqli_connect('lnmp-mysql', 'root', 'root', 'mysql');
	if ($link->connect_error) {
	    die("连接失败: " . $link->connect_error);
	} else {
			echo "连接成功";
	}
?>
```

再踩一个坑：如果站点内的某些php要求目录权限，那么nginx和php-fpm两边都得加上权限

```bash
chmod -Rf 777 ./*
```

除了mysqli，php补环境中缺失的函数。docker容器下PHP有自己特有的安装扩展方法：

* docker-php-source //在容器中创建/usr/src/php文件夹
* docker-php-ext-install //安装并启动扩展（常用）
* docker-php-ext-enable //启动PHP扩展
* docker-php-ext-configure //添加扩展自定义配置，和enable搭配使用

```bash
# 补bcmatch函数
docker-php-ext-install -j$(nproc) bcmath
# 补zip依赖
apt-get update && apt-get install -y zlib1g-dev && apt-get install -y libzip-dev
docker-php-ext-install zip
# 补gd一个图像库
apt install -y libwebp-dev libjpeg-dev libpng-dev libxpm-dev
docker-php-ext-configure gd --with-gd --with-webp-dir --with-jpeg-dir --with-png-dir --with-zlib-dir --with-xpm-dir
docker-php-ext-install gd
```

参考：****[docker安装php-gd库](https://www.cnblogs.com/xiangxisheng/p/15390194.html)、[Docker容器下PHP安装zip扩展](https://blog.csdn.net/chendongpu/article/details/120796468)**

# Dockerfile

Dockerfile用于定制镜像。在手动构建过程中，我们拉取官方的nginx镜像，需要修改配置文件、导入项目，php-fpm镜像需要额外安装mysqli等依赖函数、导入项目，Dockerfile可以将这些操作集成，并生成项目专用的镜像，以便于一键启动。

* Dockerfile

```bash
# 构建nginx镜像
FROM nginx:latest as metnginx
# 拷贝项目，目录权限
COPY default.conf /etc/nginx/conf.d/default.conf
COPY metinfo /usr/share/nginx/html/
RUN chmod -Rf 777 /usr/share/nginx/html/
WORKDIR /usr/share/nginx/html/  # nginx容器默认为根目录

EXPOSE 80

# 构建php-fpm镜像
FROM php:7.2-fpm as metphpfpm
# 拷贝项目，目录权限
COPY metinfo /var/www/html/
RUN chmod -Rf 777 /var/www/html/ \\
# 补php环境
	  && docker-php-ext-install mysqli \\
		&& docker-php-ext-install bcmath \\
		&& apt-get update \\
		&& apt-get install -y zlib1g-dev \\
		&& apt-get install -y libzip-dev \\
		&& docker-php-ext-install zip \\
		&& apt install -y libwebp-dev libjpeg-dev libpng-dev libxpm-dev libfreetype-dev \\
		&& docker-php-ext-configure gd --with-gd --with-webp-dir --with-jpeg-dir --with-png-dir --with-zlib-dir --with-xpm-dir \\
		&& docker-php-ext-install gd

EXPOSE 9000
```

* 构建：一个Dockerfile中构建多个镜像时，使用`as`设置别名，在build时使用`—target`参数

```bash
docker build --tag=metnginx --target=metnginx -f Dockerfile .
docker build --tag=metphpfpm --target=metphpfpm -f Dockerfile .
```

这样得到了两个自制镜像：metnginx和metphpfpm，mysql镜像就不需要做更改了。

* 运行：依然是创建容器，创建网络，加入网络

```bash
docker run -d -p 80:80 --name=met-nginx metnginx
docker run -d -p 9000:9000 --name=met-phpfpm metphpfpm
docker run -d -p 3306:3306 -e MYSQL_ROOT_PASSWORD=root --name=met-mysql mysql:5.7
docker network connect lnmp met-nginx
docker network connect lnmp met-phpfpm 
docker network connect lnmp met-mysql
docker restart met-nginx  # php没加入网络时运行不了
```

总结一下，手动构建过程用于调试错误，Dockfile组织好调试好的构建流程，完成项目镜像的构建。但最后运行项目仍然需要手动创建桥接网络、加入网络、启动容器一系列连通多容器的操作，这些操作同样可以被自动化完成，这就是Docker-compose、k8s等的使命。

最后再踩一个坑。以上手动构建和自动构建nginx和php-fpm都是一次性将项目文件拷贝进容器，这样带来很大的问题是一旦容器运行，修改项目文件就必须进入容器、修改、重启，而且两个容器内的站点是孤立的，每次修改还得两边改，这也就意味着php生成的js、css等静态资源并不能由nginx访问到。优雅的解决方式是，启动容器时使用`--volume` 映射到容器外的站点目录，让两个容器读取和写入同一数据卷，前面构建镜像时的COPY项目就不需要了：

```bash
docker run -d -p 9000:9000 -v "$PWD/metinfo":/var/www/html --name met-phpfpm metphpfpm
docker run -d -p 80:80 -v "$(pwd)/metinfo":/usr/share/nginx/html --name met-nginx metnginx
```

# Docker-Compose

我们先运行Dockerfile自动构建好镜像，然后运行docker-compose一键运行容器：

docker-compose.xml

```yaml
version: "3"
services:
    met-nginx:
        image: metnginx
        container_name: met-nginx
        depends_on:
            - met-phpfpm
        ports: 
            - "80:80"
        networks:
            - met
        volumes:
            - $PWD/metinfo:/usr/share/nginx/html
    met-phpfpm:
        image: metphpfpm
        container_name: met-phpfpm
        expose: 
            - "9000"
        networks:
            - met
        volumes:
            - $PWD/metinfo:/var/www/html
    met-mysql:
        image: mysql:5.7
        container_name: met-mysql
        expose: 
            - "3306"
        networks:
            - met
        environment:
            - MYSQL_ROOT_PASSWORD=roto
networks:
    met:
```

常用参数：

* image：使用镜像
* container_name：容器名称
* ports：监听端口
* networks：桥接网络
* volumes：数据卷映射
* environment：环境变量

踩坑：

1. 优先次序：depends_on将优先启动所依赖的容器
2. 端口安全：expose替代ports可以保证只能在本地访问，不会暴露在0.0.0.0
   
   ```bash
   0.0.0.0:3306->3306/tcp, :::3306->3306/tcp, 33060/tcp
   80/tcp, 0.0.0.0:9000->9000/tcp, :::9000->9000/tcp
   ```
   
   以上，mysql容器有公网3306映射，还有ipv6，还有个内网33060/tcp；php容器有内网80/tcp以及公网9000映射、ipv6。这个内网访问的33060和80端口很是迷惑，~手动启动容器完全不会产生~实际上这两个端口都是容器暴露出来仅供内网访问的
   
   那么，docker-compose.xml应该写成：expose: 9000, expose: 3306
3. networks会自动创建

