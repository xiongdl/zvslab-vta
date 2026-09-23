/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package vta.core

import vta.util.config._

/** CoreConfig.
 *
 * This is one supported configuration for VTA. This file will
 * be eventually filled out with class configurations that can be
 * mixed/matched with Shell configurations for different backends.
 */
class CoreConfig extends Config((site, here, up) => {
  case CoreKey =>
    CoreParams(
      batch = 1,
      blockOut = 16,
      blockOutFactor = 1,
      blockIn = 16,
      inpBits = 8,
      wgtBits = 8,
      uopBits = 32,
      accBits = 32,
      outBits = 8,
      uopMemDepth = 2048,
      inpMemDepth = 2048,
      wgtMemDepth = 1024,
      accMemDepth = 2048,
      outMemDepth = 2048,
      instQueueEntries = 512
    )
})

/** Core parameters normalized from the shared geometry-only VTA config. */
object GeometryCoreParams {
  private val uopBitsValue = 32

  private lazy val properties = {
    val path = sys.props.getOrElse(
      "vta.geometry.properties",
      throw new IllegalArgumentException(
        "vta.geometry.properties is required for TSIM Chisel generation"))
    val file = new java.io.File(path)
    require(file.isFile, "TSIM geometry properties do not exist: " + path)
    val loaded = new java.util.Properties()
    val input = new java.io.FileInputStream(file)
    try loaded.load(input) finally input.close()
    loaded
  }

  private def int(name: String): Int = {
    val value = properties.getProperty(name)
    require(value != null, "Missing TSIM geometry property: " + name)
    try value.toInt
    catch {
      case _: NumberFormatException =>
        throw new IllegalArgumentException(
          "Invalid TSIM geometry property " + name + "=" + value)
    }
  }

  // The shared geometry property is the VTA uop buffer capacity in bytes;
  // CoreParams stores every memory depth as a count of elements.
  private def uopMemDepthElements: Int = {
    val capacityBytes = int("UOP_MEM_DEPTH")
    require(capacityBytes % (uopBitsValue / 8) == 0,
      "TSIM uop memory capacity must be divisible by the uop size")
    capacityBytes / (uopBitsValue / 8)
  }

  def core: CoreParams = CoreParams(
    batch = int("BATCH"),
    blockOut = int("BLOCK_OUT"),
    blockOutFactor = 1,
    blockIn = int("BLOCK_IN"),
    inpBits = int("INP_BITS"),
    wgtBits = int("WGT_BITS"),
    uopBits = uopBitsValue,
    accBits = int("ACC_BITS"),
    outBits = int("OUT_BITS"),
    uopMemDepth = uopMemDepthElements,
    inpMemDepth = int("INP_MEM_DEPTH"),
    wgtMemDepth = int("WGT_MEM_DEPTH"),
    accMemDepth = int("ACC_MEM_DEPTH"),
    outMemDepth = int("OUT_MEM_DEPTH"),
    instQueueEntries = int("INST_QUEUE_ENTRIES"))
}

class GeometryCoreConfig extends Config((site, here, up) => {
  case CoreKey => GeometryCoreParams.core
})
