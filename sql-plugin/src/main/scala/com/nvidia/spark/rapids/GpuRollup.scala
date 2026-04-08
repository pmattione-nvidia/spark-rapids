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

import com.nvidia.spark.rapids.Arm.withResource

import ai.rapids.cudf.{GroupByAggregationOnColumn, Table}

import org.apache.spark.sql.rapids.aggregate.CudfAggregate
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Fused ROLLUP partial aggregate: calls libcudf rollup through
 * `com.nvidia.spark.rapids.jni.Rollup.aggregate`, which delegates to [[ai.rapids.cudf.GroupByRollup]].
 *
 * Output always includes `spark_grouping_id` after data keys (same contract as Ferdinand /
 * ExpandAggregate-style fused kernels). Partial aggregates are libcudf `GroupByAggregation`
 * requests (e.g. SUM, COUNT, MIN, MAX) passed through from [[GpuHashAggregateExec]]. Non-fused
 * plans use libcudf `Table.groupBy` in [[GpuHashAggregateExec]].
 *
 * Null keys: JNI is invoked with `ignoreNullKeys = false` (libcudf `null_policy::INCLUDE`), matching
 * `GroupByOptions.withIgnoreNullKeys(false)` on the ordinary group-by path in
 * [[GpuHashAggregateExec.performGroupByAggregation]].
 *
 * The JNI entry class `com.nvidia.spark.rapids.jni.Rollup` is loaded by reflection so this module
 * compiles against a `spark-rapids-jni` jar that may not yet publish that type. The fused rollup
 * path still requires a native JNI artifact that includes `Rollup` at runtime.
 */
object GpuRollup {

  private val jniRollupClassName = "com.nvidia.spark.rapids.jni.Rollup"

  private lazy val rollupAggregateMethod: java.lang.reflect.Method = {
    val clazz =
      try {
        Class.forName(jniRollupClassName)
      } catch {
        case e: ClassNotFoundException =>
          throw new IllegalStateException(
            "This spark-rapids-jni artifact does not include com.nvidia.spark.rapids.jni.Rollup " +
              "(needed for JNI ROLLUP fusion). Build and `mvn install` spark-rapids-jni from your " +
              "JNI tree so the dependency version matches a jar that contains Rollup.class.",
            e)
      }
    clazz.getMethods.collectFirst {
      case m
          if m.getName == "aggregate" &&
            m.getReturnType == classOf[Table] &&
            m.getParameterCount == 8 =>
        m
    }.getOrElse {
      throw new NoSuchMethodException(
        s"$jniRollupClassName.aggregate(Table,int[],int[],boolean,boolean,boolean[],boolean[]," +
          "GroupByAggregationOnColumn...)")
    }
  }

  def aggregate(
      preProcessed: ColumnarBatch,
      keySorted: Boolean,
      groupingOrdinals: Array[Int],
      rolledUpKeyIndicesAmongGroupKeys: Array[Int],
      cudfAggregates: Seq[CudfAggregate],
      aggOrdinals: Seq[Int],
      postStepDataTypes: Array[DataType]): ColumnarBatch = {
    NvtxRegistry.AGG_GROUPBY {
      withResource(GpuColumnVector.from(preProcessed)) { preProcessedTbl =>
        val cudfAggsOnColumn: Seq[GroupByAggregationOnColumn] =
          cudfAggregates.zip(aggOrdinals).map {
            case (cudfAgg, ord) => cudfAgg.groupByAggregate.onColumn(ord)
          }
        val aggsArr: Array[GroupByAggregationOnColumn] = cudfAggsOnColumn.toArray
        val emptySort: Array[Boolean] = new Array[Boolean](0)
        val aggTbl = rollupAggregateMethod
          .invoke(
            null,
            preProcessedTbl,
            groupingOrdinals,
            rolledUpKeyIndicesAmongGroupKeys,
            java.lang.Boolean.FALSE,
            java.lang.Boolean.valueOf(keySorted),
            emptySort,
            emptySort,
            aggsArr)
          .asInstanceOf[Table]
        withResource(aggTbl) { _ =>
          GpuColumnVector.from(aggTbl, postStepDataTypes)
        }
      }
    }
  }
}
