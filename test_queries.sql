SET optimizer_switch='hypergraph_optimizer=on';

DROP DATABASE IF EXISTS tmp;
CREATE DATABASE tmp;

USE tmp;

CREATE TABLE t1 (x INT);
CREATE TABLE t2 (y INT);

INSERT INTO t1 VALUES (1), (2), (3), (4), (5);
INSERT INTO t2 VALUES (2), (3), (4), (6), (7), (8), (9), (10);

CREATE TABLE t3 (z INT);
INSERT INTO t3 VALUES (1), (2), (3), (5);

EXPLAIN ANALYZE SELECT * FROM t1 join t2 ON t1.x = t2.y JOIN t3 ON t2.y = t3.z WHERE t2.y > 5 AND t3.z < 2;