---
layout: post
tags: programming
author: Chunpai
---

This is a note on Structured Query Language (SQL) for relational database systems, which could be used as a cheat-sheet for quick review of important concepts, especially for the purpose of data retrieving rather than data management and data engineering.

* TOC
{: toc}
## Some Basic Concepts 

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
|  8 | Ramesh   |  25 | Delhi     |  1500.00 |
|  9 | kaushik  |  25 | Mumbai    |  6500.00 |
| 10 | kaushik  |  25 | Delhi     |  6500.00 |
+----+----------+-----+-----------+----------+
```

1. `SELECT`: fetch certain fields of the customers available in the table:

   ```mysql
   SELECT ID, NAME, AGE FROM CUSTOMERS;
   ```

2.  `ORDER BY` : fetch the ID, NAME and SALARY fields from the CUSTOMERS table order by SALARY first, if we salaries are equal, then sort it by AGE:

	```mysql
	SELECT ID, NAME, AGE, SALARY FROM CUSTOMERS ORDER BY SALARY DESC AGE ASC;
	```

3. update the customers' address whose address was *MP* to *Pune*:

   ```mysql
   UPDATE CUSTOMERS SET ADDRESS = 'Pune' WHERE ADDRESS = 'MP';
   ```

4. find total amount of salary on each customer with same name:

   ```mysql
   SELECT NAME, SUM(SALARY) FROM CUSTOMERS GROUP BY NAME;
   ```

5. only return non-duplicate records on salary and name at the same time:

   ```mysql
   SELECT DISTINCT SALARY, NAME FROM CUSTOMERS ORDER BY SALARY;
   ```

6. fetch the rows with our own preferred order:

   ```mysql
   SELECT * FROM CUSTOMERS
      ORDER BY (CASE ADDRESS
      WHEN 'DELHI' 	 THEN 1
      WHEN 'BHOPAL' 	 THEN 2
      WHEN 'KOTA' 	 THEN 3
      WHEN 'AHMADABAD' THEN 4
      WHEN 'MP' 	THEN 5
      ELSE 100 END) ASC, ADDRESS DESC;
   ```



## Advanced Data Filtering in MySQL

1. Use `SELECT DISTINCT` to eliminate duplicate rows in a result set.

2. Use `AND` `OR` `IN` `BETWEEN` after `WHERE` for customized filtering condition.

   ```mysql
   SELECT DISTINCT * 
   FROM CUSTOMERS 
   WHERE (
   (SALARY BETWEEN 2000.0 AND 6000.0) OR (SALARY > 6500)) 
   AND (ADDRESS IN ('Kota', 'MP'));
   ```

3. Use `LIMIT` and `OFFSET` .

   > LIMIT x OFFSET y simply means skip the first y entries and then return the next x entries.
   > OFFSET can only be used with ORDER BY clause. It cannot be used on its own.
   > OFFSET value must be greater than or equal to zero.

   find the customer with 3rd highest salary:

   ```mysql
   SELECT DISTINCT * FROM CUSTOMERS ORDER BY SALARY DESC LIMIT 1 OFFSET 2;
   ```

4. Use `LIKE` for pattern match.

5. Use `IS NULL` to test whether a value is `NULL` or not.

6. Use `GROUP BY` to group rows into groups based on columns or expressions.

   > The `GROUP BY` clause must appear after the `FROM` and `WHERE` clauses. Following the `GROUP BY` keywords is a list of comma-separated columns or expressions that you want to use as criteria to group rows. MySQL evaluates the `GROUP BY` clause after the `FROM`, `WHERE` and `SELECT` clauses and before the `HAVING` , `ORDER BY` and `LIMIT` clauses. The syntax is:

   ```mysql
   SELECT 
       c1, c2,..., cn, aggregate_function(ci)
   FROM
       table
   WHERE
       where_conditions
   GROUP BY c1 , c2,...,cn;
   ```

7. Use `Having` clause to specify a filter condition for groups of rows or aggregates.

   > The `HAVING` clause is often used with the `GROUP BY` clause to filter groups based on a specified condition. If the `GROUP BY` clause is omitted, the `HAVING` clause behaves like the `WHERE` clause. MySQL evaluates the `HAVING` clause after the `FROM`, `WHERE`, `SELECT` and `GROUP BY` clauses and before `ORDER BY`, and `LIMIT` clauses.

   ```mysql
   SELECT 
       select_list
   FROM 
       table_name
   WHERE 
       search_condition
   GROUP BY 
       group_by_expression
   HAVING 
       group_condition;
   ```



## Joining Tables

1. Set table alias CUSTOMERS as C, then we can refer to the table columns using the `alias.column_name`:

   ```mysql
   SELECT C.NAME FROM CUSTOMERS AS C;
   ```

2. `INNER JOIN`: 

   > If rows from both tables cause the join condition to evaluate to `TRUE`, the `INNER JOIN` creates a new row whose columns contain all columns of rows from the tables and includes this new row in the result set. Otherwise, the `INNER JOIN` just ignores the rows.
   >
   > ![MySQL-INNER-JOIN-Venn-Diagram](/assets/img/MySQL-INNER-JOIN-Venn-Diagram.png)

   ```mysql
   SELECT column_list
   FROM table_1
   INNER JOIN table_2 ON join_condition;
   ```

3. `LEFT JOIN`: 

   > `LEFT JOIN` returns all rows from the left table regardless of whether a row from the left table has a matching row from the right table or not. If there is no match, the columns of the row from the right table will contain `NULL`.
   >
   > ![mysql-left-join-Venn-diagram](/assets/img/mysql-left-join-Venn-diagram.png)

   ```mysql
   SELECT 
       select_list
   FROM
       t1
   LEFT JOIN t2 ON 
       join_condition;
   ```

4. `RIGHT JOIN`: 

   > The `RIGHT JOIN` returns all rows from the right table regardless of having matching rows from the left table or not. It’s important to emphasize that `RIGHT JOIN` and `LEFT JOIN` clauses are functionally equivalent and they can replace each other as long as the table order is reversed.

   ```mysql
   SELECT 
       select_last
   FROM t1
   RIGHT JOIN t2 ON 
       join_condition;
   ```

5. `CROSS JOIN`:

   > The `CROSS JOIN` clause returns the Cartesian product of rows from the joined tables. Suppose you join two tables using the `CROSS JOIN` clause. The result set will include all rows from both tables, where each row is the combination of the row in the first table with the row in the second table. In general, if each table has `n` and `m` rows respectively, the result set will have `n*m` rows.

   ```mysql
   SELECT * FROM t1
   CROSS JOIN t2;
   ```

6. Self-join:  join a table to itself using table alias and connect rows within the same table using inner join and left join. 

   > The self join is often used to query hierarchical data or to compare a row with other rows within the same table. To perform a self join, you must use table aliases to not repeat the same table name twice in a single query. Note that referencing a table twice or more in a query without using table aliases will cause an error.



## Subqueries

1. Derived table - A derived table is a virtual table returned from a `SELECT` statement. 

   > Unlike a subquery, a derived table must have an alias so that you can reference its name later in the query. 

   ```mysql
   SELECT 
       column_list
   FROM
       (SELECT 
           column_list
       FROM
           table_1) derived_table_name;
   WHERE derived_table_name.c1 > 0;
   ```

   

## REFERENCE

[1] https://www.mysqltutorial.org/basic-mysql-tutorial.aspx



