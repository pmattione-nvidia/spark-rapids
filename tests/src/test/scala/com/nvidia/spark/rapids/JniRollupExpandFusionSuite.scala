/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
 * End-to-end GPU vs CPU checks for Spark ROLLUP with default plugin settings: JNI rollup expand
 * fusion is on by default ([[RapidsConf.ENABLE_JNI_ROLLUP_EXPAND_FUSION]]), and
 * [[GpuJniRollupFusionRule]] fuses when Expand does not need its internal multi-tier pre-project.
 */
class JniRollupExpandFusionSuite extends SparkQueryCompareTestSuite {

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 3 int keys sum(value)",
    createIntKeyDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1", $"cat2").agg(sum($"value").as("total"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys sum(value)",
    createIntKeyDataFrame,
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
    createIntKeyDataFrame,
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
    createIntKeyDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(count($"value").as("cnt"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys min(value)",
    createIntKeyDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(min($"value").as("mn"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys max(value)",
    createIntKeyDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(max($"value").as("mx"))
  }

  IGNORE_ORDER_testSparkResultsAreEqual(
    "JNI rollup fusion: rollup 2 int keys sum count min max(value)",
    createIntKeyDataFrame,
    repart = 2) { frame =>
    import frame.sparkSession.implicits._
    frame.rollup($"key", $"cat1").agg(
      sum($"value").as("s"),
      count($"value").as("c"),
      min($"value").as("mn"),
      max($"value").as("mx"))
  }

  private def createIntKeyDataFrame(spark: SparkSession) = {
    val schema = StructType(Seq(
      StructField("key", DataTypes.IntegerType),
      StructField("cat1", DataTypes.IntegerType),
      StructField("cat2", DataTypes.IntegerType),
      StructField("value", DataTypes.IntegerType)
    ))
    FuzzerUtils.generateDataFrame(spark, schema, 100)
  }

  private def createLongKeyDataFrame(spark: SparkSession) = {
    val schema = StructType(Seq(
      StructField("k0", DataTypes.LongType),
      StructField("k1", DataTypes.LongType),
      StructField("measure", DataTypes.LongType)
    ))
    FuzzerUtils.generateDataFrame(spark, schema, 100)
  }
}
