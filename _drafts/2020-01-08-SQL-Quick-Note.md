---
layout: post
tags: programming
author: Chunpai
---

This is a note on Structured Query Language (SQL) for relational database systems, which could be use a cheat-sheet for quick review of important concepts, especially for data analysis rather than data management and data engineering.

* TOC
{: toc}
### Some Basic Concepts 

1. What is SQL ?

   > SQL is a standard comptuer language for **relational** database management and data manipulation, which is used to *query*, *insert*, *update* and *modify* data. For the purpose data analysis, the most important part is to write efficient and correct query statements for data retrieving. 

2. What is a table, a field, a row, and a column ?

   > A table is the most common and simple form (data object) of data storage in a relational database. Every table is broken into smaller entities called fields, similar to attributes of the tables. Every column corresponds to one field, and row represents one record. 
   > 

3. What is a NULL value ?

   > A Null value in a table is a value in a field that appears to be blank. 

4. What is a Key, a Primary Key, a Composite Key, and a Foreign Key ?

   > A key is a value used to identify a record in a table uniquely. A key could be a single column or combination of multiple columns. A primary key is a *single* column value used to identify a database record uniquely, which cannot be Null and must be unique. A composite key is a primary key composed of multiple columns. A Foreign Key *references* the primary key of another table, and it helps connect different tables, and ensures rows in one table have corresponding rows in another. Foreign Key can only have values present in primary key. For example, user ID is used as primary key in user information table, and foreign key is also user ID in user transaction table. 

5. What is database normalization ?

   > Database normalization is used to organize data in an efficient way, such as eliminating redundant data or ensuring sensible data dependencies. There are 3 common database normal forms: 
   >
   > [Data Normalization Explanation]: https://www.guru99.com/database-normalization.html
   >
   > 
   >
   > 1. 1NF (First Normal Form) Rules
   >
   >    Each table cell should contain a single value, and each record needs to be unique.
   >
   > 2. 2NF (Second Normal Form) Rules
   >
   >    1NF + Single column primary key
   >
   > 3. 3NF (Third Normal Form) Rules
   >
   >    2NF + No transitive funcitonal dependencies



## Basic SQL Query Syntax

Consider the `CUSTOMERS` table having the following records:

```mysql
+----+----------+-----+-----------+----------+
| ID | NAME     | AGE | ADDRESS   | SALARY   |
+----+----------+-----+-----------+----------+
|  1 | Ramesh   |  32 | Ahmedabad |  2000.00 |
|  2 | Khilan   |  25 | Delhi     |  1500.00 |
|  3 | kaushik  |  23 | Kota      |  2000.00 |
|  4 | Chaitali |  25 | Mumbai    |  6500.00 |
|  5 | Hardik   |  27 | Bhopal    |  8500.00 |
|  6 | Komal    |  22 | MP        |  4500.00 |
|  7 | Muffy    |  24 | Indore    | 10000.00 |
+----+----------+-----+-----------+----------+
```

1. fetch certain fields of the customers available in the table:

   ```mysql
   select ID, NAME, AGE from CUSTOMERS;
   ```

2.  fetch the ID, Name and Salary fields from the CUSTOMERS table, where the salary is equal to 2000 and name is *Ramesh* or address is *Kota*, or age starts with 2:

   ```mysql
   select top 3 ID, NAME, SALARY from CUSTOMERS where SALARY = 2000 and (NAME='Ramesh' or ADDRESS = 'Kota') or AGE LIKE '2%';
   ```
   
   which is same as 
   
   ```mysql
   select ID, NAME, SALARY from CUSTOMERS where (SALARY = 2000 and (NAME='Ramesh' or ADDRESS = 'Kota')) or AGE LIKE '2%';
   ```
   
   
   
3. update the customers' address whose address was *MP* to *Pune*:

```mysql
upadte CUSTOMERS set ADDRESS = 'Pune' where ADDRESS = 'MP';
```

4. 



