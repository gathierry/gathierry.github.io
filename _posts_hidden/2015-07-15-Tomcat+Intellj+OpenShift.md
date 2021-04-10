---
layout: post
title: IntelliJ IDEA CE + Tomcat + Openshift 开发记录
tags: [web, Java, cloud]
description: 本文简单介绍了不借助Eclipse或IntelliJ专业版的情况下,以Tomcat为框架进行网络开发的配置方法，以及在Openshift平台的一种部署方法。
date: 2015-07-15
feature_image:
---

### 开发环境：

- IntelliJ IDEA CE 14
- Apache Tomcat 8.0.18

### 项目目录

	Spage(Projet)
	├─src
	│ └─java
	├─lib
	├─doc
	└─webRoot 
	  ├─WEB-INF
	  │ ├─classes
	  │ ├─lib
	  │ └─web.xml
	  ├─dist （jar、war的存放路径）
	  ├─css
	  ├─js
	  ├─view
	  └─image

### 开发流程

- 设置Java class文件输出目录。右键Module，Open module Settings，paths，将output path设为WEB-INF下的classes文件夹 **注意，实在Module的Path选项卡里修改output path**
- 将jsp-api-2.1.jar, servlet-api-2.5.jar拷贝到webRoot下。打包时需要的jar包放到lib目录。
- 右键Module，Open module Settings，Dependencies，+号引入刚刚拷贝进来的jar文件（和lib目录）
- src目录建test Package，下建Test.java

		public class Test extends HttpServlet {
			protected void doGet(HttpServletRequest req, HttpServletResponse res) throws ServletException, IOException{
				System.out.println("hellow world!");
				res.getWriter().println("Hellow world!");
			}
		}

- 在web.xml中注册该servlet

		<?xml version="1.0" encoding="ISO-8859-1"?>
		<web-app xmlns="http://xmlns.jcp.org/xml/ns/javaee"
  	   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  	   xsi:schemaLocation="http://xmlns.jcp.org/xml/ns/javaee
  	                http://xmlns.jcp.org/xml/ns/javaee/web-app_3_1.xsd"
  	   version="3.1">
  	   <servlet>
  	   		<servlet-name>test</servlet-name>
  	   		<servlet-class>test.Test</servlet-class>
  	   	</servlet>
  	   	<servlet-mapping>
  	   		<servlet-name>test</servlet-name>
  	   		<url-pattern>/test</url-pattern>
  	   	</servlet-mapping>
  	   	<session-config>
  	   		<session-timeout>30</session-timeout>
  	   	</session-config>
		</web-app>


- 右键module，compile module
- 在tomcat下 conf\catalina\localhost 创建xml文件（需要sudo）  
**注意，如果path="/hello"，则文件并也要为hello.xml**

		<?xml version="1.0"?>
		<Context path="/hello" docBase="/Users/Thierry/Desktop/Spage/Spage/webRoot" debug="0" privileged="true">
		</Context>

- webRoot下建index.html
- 浏览器localhost:8080/hello中可以看到index.html的内容  
  浏览器localhost:8080/hello/test中可以触发test.java的内容  
  
### Openshift部署

1. 添加应用
2. checkout到本地
3. 配置pom.xml文件
4. 同步代码
   以下以Spage项目为例
   
   
		projet-logiciel/
		├─src/
		│  └─main/
		│     ├─java/
		│     │  ├─db/
		│     │  ├─spider/
		│     │  └─web/
		│     │     ├─SysContextListener.java
		│     │     └─Result.java
		│     ├─resources/
		│     └─webapp/
		│         ├─css/
		│         ├─img/
		│         ├─js/
		│         ├─WEB-INF/
		│         │  ├─lib/
		│         │  ├─tld/
		│         │  └─web.xml
		│         ├─index.html
		│         └─result.jsp
		└─pom.xml
