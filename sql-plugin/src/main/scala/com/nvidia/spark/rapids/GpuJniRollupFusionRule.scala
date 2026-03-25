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

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.{Partial, PartialMerge}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.rapids.aggregate.{GpuAggregateExpression, GpuAggregateFunction, GpuSum}
import org.apache.spark.sql.types.{IntegerType, LongType}

/**
 * Fuses `GpuHashAggregateExec` over `GpuExpandExec` for Spark ROLLUP into a single aggregate
 * whose child is the expand's child, and records metadata so the first aggregation pass calls
 * `Rollup.aggregate` via JNI.
 *
 * Only the **partial** pass uses JNI rollup; **merge** still runs the generic
 * [[GpuMergeAggregateIterator]]. The partial batch includes a real
 * `spark_grouping_id` column from native code, while `groupingExpressions` on this exec is
 * rewritten to data keys only (for the pre-agg projection). `fusedRollupMergeGroupingExpressions`
 * therefore keeps the **pre-fusion** grouping list (including `spark_grouping_id`) so merge /
 * repartition keys match the partial output and distinct grouping sets are not collapsed.
 *
 * Fusion applies only when every partial aggregate is a non-distinct, unfiltered [[GpuSum]].
 * Other aggregates (e.g. [[org.apache.spark.sql.rapids.aggregate.GpuCount]]) are not yet supported
 * by the JNI result repacking path and keep the Expand + hash aggregate plan.
 *
 * Placement follows [[GpuTransitionOverrides]] (early, with other transition optimizations), not
 * the alfred fused-operator design.
 */
class GpuJniRollupFusionRule(conf: RapidsConf) extends Rule[SparkPlan] with Logging {

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!conf.isJniRollupExpandFusionEnabled) {
      return plan
    }
    plan.transformUp {
      case agg: GpuHashAggregateExec if isPartialLikeAggregate(agg) =>
        agg.child match {
          case expand: GpuExpandExec if !expand.wouldRunInternalPreProject =>
            tryFuse(expand, agg).getOrElse(agg)
          case _ => agg
        }
      case other => other
    }
  }

  private def isPartialLikeAggregate(agg: GpuHashAggregateExec): Boolean = {
    agg.aggregateExpressions.nonEmpty &&
      agg.aggregateExpressions.forall { ae =>
        ae.mode == Partial || ae.mode == PartialMerge
      }
  }

  /** JNI `Rollup.aggregate` Java wrapper expects one output column per logical aggregate (SUM). */
  private def aggregatesSupportedByJniRollup(agg: GpuHashAggregateExec): Boolean = {
    agg.aggregateExpressions.forall { e =>
      !e.isDistinct && e.filter.isEmpty && e.origAggregateFunction.isInstanceOf[GpuSum]
    }
  }

  private def isRollupGroupingMetaColumn(expr: Expression): Boolean = expr match {
    case ar: AttributeReference =>
      val n = ar.name.toLowerCase
      n == "spark_grouping_id" || n.contains("grouping_id")
    case _ => false
  }

  /**
   * Map an [[AttributeReference]] to its index in [[GpuExpandExec.output]]. Prefer `exprId`
   * (stable); fall back to **unique** name match only so anonymized plans (`none#…`) do not all
   * resolve to the first column.
   */
  private def expandOutputIndex(ar: AttributeReference, expand: GpuExpandExec): Int = {
    val byId = expand.output.indexWhere(_.exprId == ar.exprId)
    if (byId >= 0) {
      return byId
    }
    val sameName = expand.output.zipWithIndex.filter(_._1.name == ar.name)
    if (sameName.length == 1) {
      sameName.head._2
    } else {
      -1
    }
  }

  private def substituteGroupingForExpandChild(
      ne: NamedExpression,
      expand: GpuExpandExec): NamedExpression = {
    val head = expand.projections.head
    val replaced = ne.transformUp {
      case ar: AttributeReference =>
        val idx = expandOutputIndex(ar, expand)
        if (idx >= 0 && idx < head.length) {
          head(idx)
        } else {
          ar
        }
    }
    replaced match {
      case out: NamedExpression => out
      case o =>
        GpuAlias(o, ne.name)(ne.exprId, ne.qualifier)
    }
  }

  /** Rewire partial aggregate inputs from expand output attrs to [[expand.child]] (first row). */
  private def substituteAggregateForExpandChild(
      aggExp: GpuAggregateExpression,
      expand: GpuExpandExec): GpuAggregateExpression = {
    val head = expand.projections.head
    val newOrig = aggExp.origAggregateFunction.transform {
      case ar: AttributeReference =>
        val idx = expandOutputIndex(ar, expand)
        if (idx >= 0 && idx < head.length) {
          head(idx)
        } else {
          ar
        }
    }.asInstanceOf[GpuAggregateFunction]
    aggExp.copy(origAggregateFunction = newOrig)
  }

  /**
   * ROLLUP produces grouping_ids [0, 1, 3, 7, ...] = (1 << g) - 1 per projection index g.
   */
  private def isRollupGroupingIdPattern(expand: GpuExpandExec, numRollupKeyCols: Int): Boolean = {
    val projections = expand.projections
    val output = expand.output
    val gidColIndex = output.indexWhere { attr =>
      val name = attr.name.toLowerCase
      name == "spark_grouping_id" || name.contains("grouping_id")
    }
    if (gidColIndex < 0) {
      return false
    }
    val groupingIds = projections.map { proj =>
      if (gidColIndex < proj.size) {
        proj(gidColIndex) match {
          case gl: GpuLiteral if gl.dataType == LongType =>
            gl.value.asInstanceOf[Long]
          case gl: GpuLiteral if gl.dataType == IntegerType =>
            gl.value.asInstanceOf[Int].toLong
          case Literal(gid: Long, LongType) => gid
          case Literal(gid: Int, IntegerType) => gid.toLong
          case _ => -1L
        }
      } else {
        -1L
      }
    }
    groupingIds.zipWithIndex.forall { case (gid, g) =>
      gid == (1L << g) - 1
    } && groupingIds.length == numRollupKeyCols + 1
  }

  private def tryFuse(expand: GpuExpandExec, agg: GpuHashAggregateExec): Option[GpuHashAggregateExec] = {
    val dataKeyGrouping = agg.groupingExpressions.filterNot(isRollupGroupingMetaColumn)
    val gidGrouping = agg.groupingExpressions.filter(isRollupGroupingMetaColumn)
    if (gidGrouping.length != 1) {
      logInfo("[GpuJniRollupFusion] Expected exactly one spark_grouping_id-like grouping key; skipping")
      return None
    }
    val numDataKeys = dataKeyGrouping.length
    val numProj = expand.projections.length
    if (numProj != numDataKeys + 1) {
      logInfo(s"[GpuJniRollupFusion] Not a ROLLUP expand pattern: numProjections=$numProj " +
        s"!= numDataKeys+1=${numDataKeys + 1}")
      return None
    }
    if (!isRollupGroupingIdPattern(expand, numDataKeys)) {
      logInfo("[GpuJniRollupFusion] Expand grouping_id literals do not match ROLLUP pattern; skipping")
      return None
    }
    if (!aggregatesSupportedByJniRollup(agg)) {
      logInfo("[GpuJniRollupFusion] Partial aggregates are not all supported SUM (non-distinct, " +
        "no filter); skipping JNI rollup fusion")
      return None
    }
    val rolledSuffixLen = numProj - 1
    val firstRoll = numDataKeys - rolledSuffixLen
    if (firstRoll < 0 || firstRoll > numDataKeys) {
      return None
    }
    val rolledAmongDataKeys: Seq[Int] = (firstRoll until numDataKeys).toSeq

    val newGrouping = dataKeyGrouping.map(substituteGroupingForExpandChild(_, expand))
    val newAggs = agg.aggregateExpressions.map(substituteAggregateForExpandChild(_, expand))
    val gidAttr = gidGrouping.head.toAttribute

    logInfo(s"[GpuJniRollupFusion] Fusing Expand into partial GpuHashAggregateExec: " +
      s"dataKeys=$numDataKeys rolledAmongDataKeys=${rolledAmongDataKeys.mkString("[", ",", "]")}")

    // Merge pass must group by data keys + spark_grouping_id; see class doc above.
    Some(agg.copy(
      child = expand.child,
      groupingExpressions = newGrouping,
      aggregateExpressions = newAggs,
      fusedRollupRolledUpKeyOrdinals = Some(rolledAmongDataKeys),
      fusedRollupGroupingIdAttribute = Some(gidAttr),
      fusedRollupMergeGroupingExpressions = Some(agg.groupingExpressions)))
  }
}
