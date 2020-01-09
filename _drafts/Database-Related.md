---
layout: post
tags: reinforcement-learning
author: Chunpai
---

This is a quick note on manipulating different database files, such as .mdb .sql files. 





* TOC
{: toc}
We can convert the .mdb file into .sql file, and then use mysql to access it. First, we need to install the mysql server:

```
sudo apt-get update
sudo apt-get install mysql-server
sudo ufw allow mysql  #allow remote access
systemctl start mysql #start the MySQL service
systemctl enable mysql #launch at reboot
/usr/bin/mysql -u root -p 

```

We can set the root password by replacing the *password* with new password:

```
UPDATE mysql.user SET authentication_string = PASSWORD('password') WHERE User = 'root';
FLUSH PRIVILEGES;
SELECT User, Host, authentication_string FROM mysql.user;
```

In addition, we need to create an empty dataset:

```
mysql> CREATE DATABASE myDB;
```

Now we can convert the .mdb file to .sql file:

```
sudo apt-get install mdbtools
mdb-tables database.mdb
```

Once we are satisfied that the appropriate tables are there we can generate a schema to be used with MySQL. For this we will use the ‘mdb-schema’ command. mdb-schema produces data definition language output for the given database. This can be further passed to another database engine (MySQL, Oracle, Sybase, PostgreSQL etc) to create a replica of the original access table format.

```
mdb-schema table_name database.mdb > file.sql
```

The above will generate the schema for all the tables in the ‘database.mdb’ database and save it to the schema.sql file. If you want to output the schema for a particular table rather than all the tables, you will need to use the ‘-T’ flag. The ‘-T’ flag will restrict the schema for a particular table.

```
mdb-schema -T table_name database.mdb > file.sql
mysql -u username -p myDB < file.sql
```

Rather than saving the schema to a file we can directly import it into MySQL. The following will directly create a MySQL table using the schema from the MDB database by piping the output of mdb-schema to MySQL.

```
mdb-schema database.mdb mysql | sed "s/^-/#/" | grep -v ^DROP | mysql -u username --password=PASSWORD database_name 
```

Next, we need to install the python SQL module *pymysql*:

```
pip install pymysql
```

In addition, we can convert the .mdb into .csv file, and use pandas or csv module to process it in python.



