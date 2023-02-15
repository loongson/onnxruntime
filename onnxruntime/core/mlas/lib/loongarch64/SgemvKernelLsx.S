/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemvKernelLsx.s

Abstract:

    This module implements the kernels for the single precision matrix/vector
    multiply operation (SGEMV).

--*/

#include "asmmacro.h"

        .text

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows. This handles the special case of M=1.

    The elements in matrix B are not transposed.

Arguments:

    A (x0) - Supplies the address of matrix A.

    B (x1) - Supplies the address of matrix B.

    C (x2) - Supplies the address of matrix C.

    CountK (x3) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountN (x4) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldb (x5) - Supplies the first dimension of matrix B.

    ZeroMode (x6) - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasGemvFloatKernel

	addi.d  $r19,   $r8,    -64
	blt	$r19,	$zero,	.LSgemvN.ProcessRemainingCountN
	add.d   $r17,   $r4,    $zero

//
// Process 64 columns at a time in a loop.
//

.LSgemvN.ProcessColumnLoopBy64:
	vld     $vr4,   $r5,    0
	addi.d	$r18,	$r5,	256
	vld	$vr5,	$r5,	16
	andi	$r19,	$r10,	0xFF
	add.d	$r16,	$r7,	$zero
	vld	$vr6,	$r5,	32
	beqz	$r19,	.LSgemvN.LoadOutputBy64
	vsub.w  $vr16, $vr16, $vr16
	vsub.w  $vr17, $vr17, $vr17
	vsub.w  $vr18, $vr18, $vr18
	vsub.w  $vr19, $vr19, $vr19
	vsub.w  $vr20, $vr20, $vr20
	vsub.w  $vr21, $vr21, $vr21
	vsub.w  $vr22, $vr22, $vr22
	vsub.w  $vr23, $vr23, $vr23
	vsub.w  $vr24, $vr24, $vr24
	vsub.w  $vr25, $vr25, $vr25
	vsub.w  $vr26, $vr26, $vr26
	vsub.w  $vr27, $vr27, $vr27
	vsub.w  $vr28, $vr28, $vr28
	vsub.w  $vr29, $vr29, $vr29
	vsub.w  $vr30, $vr30, $vr30
	vsub.w  $vr31, $vr31, $vr31
        b       .LSgemvN.MultiplyAccumulateBy64

.LSgemvN.LoadOutputBy64:
	vld	$vr16,	$r6,	0
	vld	$vr17,	$r6,	16
	vld	$vr18,	$r6,	32
	vld	$vr19,	$r6,	32+16
	vld	$vr20,	$r6,	64
	vld	$vr21,	$r6,	64+16
	vld	$vr22,	$r6,	96
	vld	$vr23,	$r6,	96+16
	vld	$vr24,	$r6,	128
	vld	$vr25,	$r6,	128+16
	vld	$vr26,	$r6,	160
	vld	$vr27,	$r6,	160+16
	vld	$vr28,	$r6,	192
	vld	$vr29,	$r6,	192+16
	vld	$vr30,	$r6,	224
	vld	$vr31,	$r6,	224+16

.LSgemvN.MultiplyAccumulateBy64:
	vldrepl.w	$vr0,	$r4,	0
	addi.d	$r4,	$r4,	4
	addi.d	$r16,	$r16,	-1
	vfmadd.s	$vr16,	$vr4,	$vr0,	$vr16
	vld	$vr7,	$r5,	48
	vfmadd.s	$vr17,	$vr5,	$vr0,	$vr17
	vld	$vr4,	$r5,	64
	vfmadd.s	$vr18,	$vr6,	$vr0,	$vr18
	vld	$vr5,	$r5,	80
	vfmadd.s	$vr19,	$vr7,	$vr0,	$vr19
	vld	$vr6,	$r5,	96
	vfmadd.s	$vr20,	$vr4,	$vr0,	$vr20
	vld	$vr7,	$r5,	112
	vfmadd.s	$vr21,	$vr5,	$vr0,	$vr21
	vld	$vr4,	$r5,	128
	vfmadd.s	$vr22,	$vr6,	$vr0,	$vr22
	vld	$vr5,	$r5,	144
	vfmadd.s	$vr23,	$vr7,	$vr0,	$vr23
	vld	$vr6,	$r5,	160
	vfmadd.s	$vr24,	$vr4,	$vr0,	$vr24
	vld	$vr7,	$r5,	176
	vfmadd.s	$vr25,	$vr5,	$vr0,	$vr25
	vld	$vr4,	$r5,	192
	vfmadd.s	$vr26,	$vr6,	$vr0,	$vr26
	vld	$vr5,	$r5,	208
	vfmadd.s	$vr27,	$vr7,	$vr0,	$vr27
	vld	$vr6,	$r5,	224
	vfmadd.s	$vr28,	$vr4,	$vr0,	$vr28
	vld	$vr7,	$r5,	240
	alsl.d  $r5,   $r9,   $r5,    2
	beqz	$r16,	.LSgemvN.StoreOutputBy64
	vld	$vr4,	$r5,	0
	vfmadd.s	$vr29,	$vr5,	$vr0,	$vr29
	vld	$vr5,	$r5,	16
	vfmadd.s	$vr30,	$vr6,	$vr0,	$vr30
	vld	$vr6,	$r5,	32
	vfmadd.s	$vr31,	$vr7,	$vr0,	$vr31
        b       .LSgemvN.MultiplyAccumulateBy64

.LSgemvN.StoreOutputBy64:
	vst     $vr16,	$r6,    0
	vst	$vr17,	$r6,	16
	vfmadd.s	$vr29,	$vr5,	$vr0,	$vr29
	vst	$vr18,	$r6,	32
	vst	$vr19,	$r6,	32+16
	vfmadd.s	$vr30,	$vr6,	$vr0,	$vr30
	vst	$vr20,	$r6,	64
	vst	$vr21,	$r6,	64+16
	vfmadd.s	$vr31,	$vr7,	$vr0,	$vr31
	vst	$vr22,	$r6,	96
	vst	$vr23,	$r6,	96+16
	addi.d	$r8,	$r8,	-64
	vst	$vr24,	$r6,	128
	vst	$vr25,	$r6,	128+16
	add.d	$r4,	$r17,	$zero
	vst	$vr26,	$r6,	160
	vst	$vr27,	$r6,	160+16
	add.d	$r5,	$r18,	$zero
	vst	$vr28,	$r6,	192
	vst	$vr29,	$r6,	192+16
	vst	$vr30,	$r6,	224
	vst	$vr31,	$r6,	224+16
	addi.d	$r6,	$r6,	256
	beq	$r8,	$zero,	.LSgemvN.ExitKernel
	addi.d  $r19,    $r8,    -64
	bge	$r19,	$zero,	.LSgemvN.ProcessColumnLoopBy64

//
// Process the remaining 1 to 63 columns.
//

.LSgemvN.ProcessRemainingCountN:
        andi     $r19,   $r10,   0xFF
	beq	$r19,	$zero,	.LSgemvN.LoadOutputPartial32
	vsub.w  $vr16, $vr16, $vr16
	vsub.w  $vr17, $vr17, $vr17
	vsub.w  $vr18, $vr18, $vr18
	vsub.w  $vr19, $vr19, $vr19
	vsub.w  $vr20, $vr20, $vr20
	vsub.w  $vr21, $vr21, $vr21
	vsub.w  $vr22, $vr22, $vr22
	vsub.w  $vr23, $vr23, $vr23
	vsub.w  $vr24, $vr24, $vr24
	vsub.w  $vr25, $vr25, $vr25
	vsub.w  $vr26, $vr26, $vr26
	vsub.w  $vr27, $vr27, $vr27
	vsub.w  $vr28, $vr28, $vr28
	vsub.w  $vr29, $vr29, $vr29
	vsub.w  $vr30, $vr30, $vr30
	vsub.w  $vr31, $vr31, $vr31
	vsub.w  $vr1, $vr1, $vr1
        b       .LSgemvN.ProcessNextPartialRow

.LSgemvN.LoadOutputPartial32:
	add.d	$r18,	$r6,	$zero
        andi     $r19,   $r8,   0x20
	beqz	$r19,	.LSgemvN.LoadOutputPartial16
	vld	$vr16,	$r18,	0
	vld	$vr17,	$r18,	16
	addi.d	$r18,	$r18,	128
	vld	$vr18,	$r18,	-96
	vld	$vr19,	$r18,	-96+16
	vld	$vr20,	$r18,	-64
	vld	$vr21,	$r18,	-64+16
	vld	$vr22,	$r18,	-32
	vld	$vr23,	$r18,	-32+16

.LSgemvN.LoadOutputPartial16:
        andi     $r19,   $r8,   0x10
        beqz    $r19,	.LSgemvN.LoadOutputPartial8
	vld	$vr24,	$r18,	0
	vld	$vr25,	$r18,	16
	addi.d	$r18,	$r18,	64
	vld	$vr26,	$r18,	-32
	vld	$vr27,	$r18,	-32+16

.LSgemvN.LoadOutputPartial8:
	andi     $r19,   $r8,   0x8
	beqz    $r19,	.LSgemvN.LoadOutputPartial4
	vld	$vr28,	$r18,	0
	vld	$vr29,	$r18,	16
	addi.d	$r18,	$r18,	32

.LSgemvN.LoadOutputPartial4:
        andi     $r19,    $r8,    0x4
        beqz    $r19,	.LSgemvN.LoadOutputPartial2
	vld	$vr30,	$r18,	0
	addi.d	$r18,	$r18,	16

.LSgemvN.LoadOutputPartial2:
        andi     $r19,    $r8,    0x2
        beqz    $r19,	.LSgemvN.LoadOutputPartial1
	fld.d	$f31,	$r18,	0
	addi.d	$r18,	$r18,	8

.LSgemvN.LoadOutputPartial1:
        andi     $r19,    $r8,    0x1
        beqz    $r19,	.LSgemvN.ProcessNextPartialRow
	fld.s	$f1,	$r18,	0

.LSgemvN.ProcessNextPartialRow:
	vldrepl.w       $vr0,   $r4,    0
	addi.d	$r4,	$r4,	4
	addi.d	$r7,	$r7,	-1
	add.d	$r18,	$r5,	$zero

.LSgemvN.MultiplyAccumulatePartial32:
        andi     $r19,   $r8,   0x20
        beqz    $r19,	.LSgemvN.MultiplyAccumulatePartial16
	vld	$vr4,	$r18,	0
	vld	$vr5,	$r18,	16
	addi.d	$r18,	$r18,	128
	vfmadd.s	$vr16,	$vr4,	$vr0,	$vr16
	vld	$vr6,	$r18,	-96
	vld	$vr7,	$r18,	-96+16
	vfmadd.s	$vr17,	$vr5,	$vr0,	$vr17
	vld	$vr4,	$r18,	-64
	vld	$vr5,	$r18,	-64+16
	vfmadd.s	$vr18,	$vr6,	$vr0,	$vr18
	vfmadd.s	$vr19,	$vr7,	$vr0,	$vr19
	vld	$vr6,	$r18,	-32
	vld	$vr7,	$r18,	-32+16
	vfmadd.s	$vr20,	$vr4,	$vr0,	$vr20
	vfmadd.s	$vr21,	$vr5,	$vr0,	$vr21
	vfmadd.s	$vr22,	$vr6,	$vr0,	$vr22
	vfmadd.s	$vr23,	$vr7,	$vr0,	$vr23

.LSgemvN.MultiplyAccumulatePartial16:
        andi     $r19,   $r8,   0x10
        beqz    $r19,	.LSgemvN.MultiplyAccumulatePartial8
	vld	$vr4,	$r18,	0
	vld	$vr5,	$r18,	16
	addi.d	$r18,	$r18,	64
	vfmadd.s	$vr24,	$vr4,	$vr0,	$vr24
	vld	$vr6,	$r18,	-32
	vld	$vr7,	$r18,	-32+16
	vfmadd.s	$vr25,	$vr5,	$vr0,	$vr25
	vfmadd.s	$vr26,	$vr6,	$vr0,	$vr26
	vfmadd.s	$vr27,	$vr7,	$vr0,	$vr27

.LSgemvN.MultiplyAccumulatePartial8:
        andi     $r19,   $r8,   0x8
        beqz    $r19,	.LSgemvN.MultiplyAccumulatePartial4
	vld	$vr4,	$r18,	0
	vld	$vr5,	$r18,	16
	addi.d	$r18,	$r18,	32
	vfmadd.s	$vr28,	$vr4,	$vr0,	$vr28
	vfmadd.s	$vr29,	$vr5,	$vr0,	$vr29

.LSgemvN.MultiplyAccumulatePartial4:
        andi     $r19,   $r8,   0x4
        beqz    $r19,	.LSgemvN.MultiplyAccumulatePartial2
	vld	$vr4,	$r18,	0
	addi.d	$r18,	$r18,	16
	vfmadd.s	$vr30,	$vr4,	$vr0,	$vr30

.LSgemvN.MultiplyAccumulatePartial2:
        andi     $r19,   $r8,   0x2
        beqz    $r19,	.LSgemvN.MultiplyAccumulatePartial1
	fld.d	$f4,	$r18,	0
	addi.d	$r18,	$r18,	8
	vfmadd.s	$vr31,	$vr4,	$vr0,	$vr31

.LSgemvN.MultiplyAccumulatePartial1:
        andi     $r19,   $r8,   0x1
        beqz    $r19,	.LSgemvN.AdvancePartialRow
	fld.s	$f4,	$r18,	0
	vfmadd.s	$vr1,	$vr4,	$vr0,	$vr1

.LSgemvN.AdvancePartialRow:
	alsl.d	$r5,	$r9,	$r5,	2
	bnez	$r7,	.LSgemvN.ProcessNextPartialRow

.LSgemvN.StoreOutputPartial32:
        andi     $r19,   $r8,   0x20
        beqz    $r19,	.LSgemvN.StoreOutputPartial16
	vst	$vr16,	$r6,	0
	vst	$vr17,	$r6,	16
	addi.d	$r6,	$r6,	128
	vst	$vr18,	$r6,	-96
	vst	$vr19,	$r6,	-96+16
	vst	$vr20,	$r6,	-64
	vst	$vr21,	$r6,	-64+16
	vst	$vr22,	$r6,	-32
	vst	$vr23,	$r6,	-32+16

.LSgemvN.StoreOutputPartial16:
        andi     $r19,   $r8,   0x10
        beqz    $r19,	.LSgemvN.StoreOutputPartial8
	vst	$vr24,	$r6,	0
	vst	$vr25,	$r6,	16
	addi.d	$r6,	$r6,	64
	vst	$vr26,	$r6,	-32
	vst	$vr27,	$r6,	-32+16

.LSgemvN.StoreOutputPartial8:
        andi     $r19,   $r8,   0x8
        beqz    $r19,	.LSgemvN.StoreOutputPartial4
	vst	$vr28,	$r6,	0
	vst	$vr29,	$r6,	16
	addi.d	$r6,	$r6,	32

.LSgemvN.StoreOutputPartial4:
        andi     $r19,   $r8,   0x4
        beqz    $r19,	.LSgemvN.StoreOutputPartial2
	vst	$vr30,	$r6,	0
	addi.d	$r6,	$r6,	16

.LSgemvN.StoreOutputPartial2:
        andi     $r19,   $r8,   0x2
        beqz    $r19,	.LSgemvN.StoreOutputPartial1
	fst.d	$f31,	$r6,	0
	addi.d	$r6,	$r6,	8

.LSgemvN.StoreOutputPartial1:
        andi     $r19,   $r8,   0x1
        beqz    $r19,	.LSgemvN.ExitKernel
	fst.s	$f1,	$r6,	0

.LSgemvN.ExitKernel:
	jirl    $r0,  $r1,    0

        .end
