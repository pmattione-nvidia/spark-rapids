/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.spark.rapids

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.rapids.shims.TrampolineConnectShims._
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}

/**
 * GPU vs CPU checks for Expand-related plans (cube, rollup, grouping sets) and plain groupBy.
 * When `spark.rapids.sql.jniRollupExpandFusion.enabled` is true and the plan matches
 * ([[GpuJniRollupFusionRule]]), partial rollup aggregates can use JNI / libcudf rollup; other
 * cases still exercise [[GpuExpandExec]] with the generic aggregate path.
 */
class ExpandExecSuite extends SparkQueryCompareTestSuite {

  IGNORE_ORDER_testSparkResultsAreEqual("group with aggregates",
    createDataFrame, repart = 2) {
    // There are only 100 integer values in the data generated so we don't need
    // to worry about an overflow in SUM
    frame => {
      import frame.sparkSession.implicits._
      frame.groupBy($"key")
        .agg(
          countDistinct($"cat1").as("cat1_cnt_distinct"),
          countDistinct($"cat2").as("cat2_cnt_distinct"),
          count($"cat1").as("cat2_cnt"),
          count($"cat2").as("cat2_cnt"),
          sum($"value").as("total"))
    }
  }

  IGNORE_ORDER_testSparkResultsAreEqual("cube with count",
    createDataFrame, repart = 2) {
    frame => {
      import frame.sparkSession.implicits._
      frame.cube($"key", $"cat1", $"cat2").count()
    }
  }

  IGNORE_ORDER_testSparkResultsAreEqual("cube with count distinct",
    createDataFrame, repart = 2) {
    frame => {
      import frame.sparkSession.implicits._
      frame.rollup($"key", $"cat2")
        .agg(countDistinct($"cat1").as("cat1_cnt"))
    }
  }

  IGNORE_ORDER_testSparkResultsAreEqual("cube with sum",
    createDataFrame, repart = 2) {
    // There are only 100 integer values in the data generated so we don't need
    // to worry about an overflow in SUM
    frame => {
      import frame.sparkSession.implicits._
      frame.cube($"key", $"cat1", $"cat2").sum()
    }
  }

  IGNORE_ORDER_testSparkResultsAreEqual("rollup with count",
    createDataFrame, repart = 2) {
    frame => {
      import frame.sparkSession.implicits._
      frame.rollup($"key", $"cat1", $"cat2").count()
    }
  }

  IGNORE_ORDER_testSparkResultsAreEqual("rollup with count distinct",
    createDataFrame, repart = 2) {
    frame => {
      import frame.sparkSession.implicits._
      frame.rollup($"key", $"cat2")
        .agg(countDistinct($"cat1").as("cat1_cnt"))
    }
  }

  IGNORE_ORDER_testSparkResultsAreEqual("rollup with sum",
    createDataFrame, repart = 2) {
    frame => {
      import frame.sparkSession.implicits._
      frame.rollup($"key", $"cat1", $"cat2").sum()
    }
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 3 int keys sum(value)",
    createDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1", $"cat2").agg(sum($"value").as("total"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys sum(value)",
    createDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(sum($"value").as("total"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 long keys sum(measure)",
    createLongKeyDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"k0", $"k1").agg(sum($"measure").as("total"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: SQL GROUP BY ... WITH ROLLUP sum",
    createDataFrame,
    repart = 2) { frame =>
    frame.createOrReplaceTempView("t_jni_rollup_fusion")
    val sql =
      """SELECT key, cat1, cat2, SUM(value) AS total
        |FROM t_jni_rollup_fusion
        |GROUP BY key, cat1, cat2 WITH ROLLUP""".stripMargin
    frame.sparkSession.sql(sql)
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys count(value)",
    createDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(count($"value").as("cnt"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys min(value)",
    createDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(min($"value").as("mn"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys max(value)",
    createDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(max($"value").as("mx"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys sum count min max(value)",
    createDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(
      sum($"value").as("s"),
      count($"value").as("c"),
      min($"value").as("mn"),
      max($"value").as("mx"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual("sql with grouping expressions",
    createDataFrame, repart = 2) {
    frame => {
      frame.createOrReplaceTempView("t0")
      val sql =
        """SELECT key, cat1, cat2, COUNT(DISTINCT value)
      FROM t0
      GROUP BY key, cat1, cat2
      GROUPING SETS ((key, cat1), (key, cat2))""".stripMargin
      frame.sparkSession.sql(sql)
    }
  }

  IGNORE_ORDER_testSparkResultsAreEqual("sql with different shape " +
    "grouping expressions", createDataFrame, repart = 2) {
    frame => {
      frame.createOrReplaceTempView("t0")
      val sql =
        """SELECT key, cat1, cat2, COUNT(DISTINCT value)
      FROM t0
      GROUP BY key, cat1, cat2
      GROUPING SETS ((key, cat1), (key, cat2), (cat1, cat2), cat1, cat2)""".stripMargin
      frame.sparkSession.sql(sql)
    }
  }

  private def createDataFrame(spark: SparkSession): DataFrame = {
    val schema = StructType(Seq(
      StructField("key", DataTypes.IntegerType),
      StructField("cat1", DataTypes.IntegerType),
      StructField("cat2", DataTypes.IntegerType),
      StructField("value", DataTypes.IntegerType)
    ))
    FuzzerUtils.generateDataFrame(spark, schema, 100)
  }

  private def createLongKeyDataFrame(spark: SparkSession): DataFrame = {
    val schema = StructType(Seq(
      StructField("k0", DataTypes.LongType),
      StructField("k1", DataTypes.LongType),
      StructField("measure", DataTypes.LongType)
    ))
    FuzzerUtils.generateDataFrame(spark, schema, 100)
  }
}