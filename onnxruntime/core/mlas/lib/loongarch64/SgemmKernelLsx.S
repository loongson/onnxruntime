/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelLsx.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/

#include "asmmacro.h"

        .text

//
// ClearRowAccumulators
//
// Generates the code to clear the accumulators for a single row of the output
// block.
//

        .macro  ClearRowAccumulators Columns, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

	vsub.b	$vr\Vec1Reg\(),	$vr\Vec1Reg\(),	$vr\Vec1Reg\()
	vsub.b	$vr\Vec2Reg\(),	$vr\Vec2Reg\(),	$vr\Vec2Reg\()
.if \Columns\() > 8
	vsub.b	$vr\Vec3Reg\(),	$vr\Vec3Reg\(),	$vr\Vec3Reg\()
	vsub.b	$vr\Vec4Reg\(),	$vr\Vec4Reg\(),	$vr\Vec4Reg\()
.endif

        .endm


//
// ClearBlockAccumulators
//
// Generates the code to clear the accumulators for a single row of the output
// block.
//

        .macro  ClearBlockAccumulators Columns, Rows

        ClearRowAccumulators \Columns\(),16,17,18,19
.if \Rows\() >= 2
        ClearRowAccumulators \Columns\(),20,21,22,23
.endif
.if \Rows\() >= 4
        ClearRowAccumulators \Columns\(),24,25,26,27
        ClearRowAccumulators \Columns\(),28,29,30,31
.endif

        .endm

//
// LoadMatrixAElementsBy4
// LoadMatrixAElementsBy1
//
// Generates the code to load 1 or 4 elements from matrix A.
//

        .macro  LoadMatrixAElementsBy4 Rows

	vld	$vr8,	$r4,	0
	addi.d	$r4,	$r4,	16
.if \Rows\() >= 2
	vld	$vr9,	$r13,	0
	addi.d	$r13,	$r13,	16
.endif
.if \Rows\() >= 4
	vld	$vr10,	$r14,	0
	addi.d	$r14,	$r14,	16
	vld	$vr11,	$r15,	0
	addi.d	$r15,	$r15,	16
.endif

        .endm

        .macro  LoadMatrixAElementsBy1 Rows

	fld.s	$f8,	$r4,	0
	addi.d	$r4,	$r4,	4
.if \Rows\() >= 2
	fld.s	$f9,	$r13,	0
	addi.d	$r13,	$r13,	4
.endif
.if \Rows\() >= 4
	fld.s	$f10,	$r14,	0
	addi.d	$r14,	$r14,	4
	fld.s	$f11,	$r15,	0
	addi.d	$r15,	$r15,	4
.endif

        .endm

//
// MultiplyAccumulateRow
//
// Generates the code to multiply and accumulate a single row of the output
// block.
//

        .macro  MultiplyAccumulateRow Columns, MatrixAReg, Broadcast, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

	vreplvei.w	$vr12,	$\MatrixAReg\(),	\Broadcast\()
	vfmadd.s	$vr\Vec1Reg\(),	$vr4,	$vr12,	$vr\Vec1Reg\()
	vfmadd.s        $vr\Vec2Reg\(),	$vr5,	$vr12,	$vr\Vec2Reg\()
.if \Columns\() > 8
	vreplvei.w      $vr12,   $\MatrixAReg\(),        \Broadcast\()
	vfmadd.s        $vr\Vec3Reg\(),	$vr6,	$vr12,	$vr\Vec3Reg\()
	vfmadd.s        $vr\Vec4Reg\(),	$vr7,	$vr12,	$vr\Vec4Reg\()
.endif

        .endm

//
// MultiplyAccumulateBlock
//
// Generates the code to multiply and accumulate into the output block.
//

        .macro  MultiplyAccumulateBlock Columns, Rows, Broadcast

	MultiplyAccumulateRow \Columns\(),vr8,\Broadcast\(),16,17,18,19
.if \Rows\() >= 2
	MultiplyAccumulateRow \Columns\(),vr9,\Broadcast\(),20,21,22,23
.endif
.if \Rows\() >= 4
	MultiplyAccumulateRow \Columns\(),vr10,\Broadcast\(),24,25,26,27
	MultiplyAccumulateRow \Columns\(),vr11,\Broadcast\(),28,29,30,31
.endif

        .endm

//
// ComputeBlockLoop
//
// Generates the code to loop over K entries of the input matrices to produce
// the output block.
//

        .macro  ComputeBlockLoop Mode, Columns, Rows

        ClearBlockAccumulators \Columns\(),\Rows\()

.if \Rows\() >= 2
	alsl.d	$r13,	$r10,	$r4,	2
.endif
.if \Rows\() >= 4
	alsl.d	$r14,	$r10,	$r13,	2
	alsl.d	$r15,	$r10,	$r14,	2
.endif

	addi.d	$r12,	$r7,	-4
	srli.d	$r19,	$r12,	63
	bnez	$r19, .L\Mode\().ProcessRemaining\Columns\().x\Rows\().Blocks

.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4Loop:
        LoadMatrixAElementsBy4 \Rows\()
	vld	$vr4,	$r5,	0
	vld	$vr5,	$r5,	16
	addi.d	$r5,	$r5,	64*4
.if \Columns\() > 8
	vld	$vr6,	$r5,	-56*4
	vld	$vr7,	$r5,	-56*4+16
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),0
	vld	$vr4,	$r5,	-48*4
	vld	$vr5,	$r5,	-48*4+16
.if \Columns\() > 8
	vld	$vr6,	$r5,	-40*4
	vld	$vr7,	$r5,	-40*4+16
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),1
	vld	$vr4,	$r5,	-32*4
	vld	$vr5,	$r5,	-32*4+16
.if \Columns\() > 8
        vld     $vr6,   $r5,    -24*4
        vld     $vr7,   $r5,    -24*4+16
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),2
        vld     $vr4,   $r5,    -16*4
        vld     $vr5,   $r5,    -16*4+16
.if \Columns\() > 8
        vld     $vr6,   $r5,    -8*4
        vld     $vr7,   $r5,    -8*4+16
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),3
	addi.d	$r12,	$r12,	-4
	srli.d  $r19,    $r12,    63
	beqz	$r19,	.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4Loop

.L\Mode\().ProcessRemaining\Columns\().x\Rows\().Blocks:
	addi.d	$r12,	$r12,	4
	beqz    $r12,	.L\Mode\().Output\Columns\().x\Rows\().Block

.L\Mode\().Compute\Columns\().x\Rows\().BlockBy1Loop:
        LoadMatrixAElementsBy1 \Rows\()
	vld     $vr4,   $r5,    0
        vld     $vr5,   $r5,    16
        addi.d  $r5,    $r5,    16*4
.if \Columns\() > 8
        vld     $vr6,   $r5,    -8*4
        vld     $vr7,   $r5,    -8*4+16
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),0
	addi.d  $r12,    $r12,    -1
	bnez	$r12,	.L\Mode\().Compute\Columns\().x\Rows\().BlockBy1Loop

.L\Mode\().Output\Columns\().x\Rows\().Block:

        .endm

//
// MultiplyAlphaRow
//
// Generates the code to multiply a single row of the output block by the alpha
// value.
//

        .macro  MultiplyAlphaRow Columns, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.if \Columns\() <= 4
	vreplvei.w	$vr13,	$vr0,	0
	vfmul.s	$vr\Vec1Reg\(),	$vr\Vec1Reg\(),	$vr13
.elif \Columns\() <= 8
	vreplvei.w      $vr13,   $vr0,   0
	vfmul.s	$vr\Vec1Reg\(), $vr\Vec1Reg\(),	$vr13
	vfmul.s $vr\Vec2Reg\(), $vr\Vec2Reg\(),	$vr13
.elif \Columns\() <= 12
	vreplvei.w      $vr13,   $vr0,   0
	vfmul.s $vr\Vec1Reg\(), $vr\Vec1Reg\(),	$vr13
	vfmul.s $vr\Vec2Reg\(), $vr\Vec2Reg\(),	$vr13
	vfmul.s $vr\Vec3Reg\(), $vr\Vec3Reg\(),	$vr13
.else
	vreplvei.w      $vr13,   $vr0,   0
	vfmul.s $vr\Vec1Reg\(), $vr\Vec1Reg\(),	$vr13
	vfmul.s $vr\Vec2Reg\(), $vr\Vec2Reg\(),	$vr13
	vfmul.s $vr\Vec3Reg\(), $vr\Vec3Reg\(),	$vr13
	vfmul.s $vr\Vec4Reg\(), $vr\Vec4Reg\(),	$vr13
.endif

        .endm

//
// MultiplyAlphaBlock
//
// Generates the code to multiply the output block by the alpha value.
//

        .macro  MultiplyAlphaBlock Columns, Rows

        MultiplyAlphaRow \Columns\(),16,17,18,19
.if \Rows\() >= 2
        MultiplyAlphaRow \Columns\(),20,21,22,23
.endif
.if \Rows\() >= 4
        MultiplyAlphaRow \Columns\(),24,25,26,27
        MultiplyAlphaRow \Columns\(),28,29,30,31
.endif

        .endm

//
// OutputRow1Element
// OutputRow2Element
// OutputRow4Element
// OutputRow8Element
// OutputRow16Element
//
// Generates the code to store elements to the output block.
//

        .macro  OutputRow1Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
	fld.s	$f4,	$\AddrReg\(),	0
	vreplvei.w    $vr1,   $vr0,   0
	vpickve2gr.du    $r19,   $vr4,   1
	vfmadd.s	$vr4,	$vr\Vec1Reg\(),	$vr1,	$vr4
	vinsgr2vr.d	$vr4,	$r19,	1
	fst.s	$f4,	$\AddrReg\(),	0
.else
	fst.s	$f\Vec1Reg\(),	$\AddrReg\(),	0
.endif

        .endm

        .macro  OutputRow2Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
	fld.d	$f4,	$\AddrReg\(),	0
        vreplvei.w    $vr14,   $vr0,   0
	vpickve2gr.du    $r19,   $vr4,   1
        vfmadd.s        $vr4,   $vr\Vec1Reg\(), $vr14,   $vr4
        vinsgr2vr.d     $vr4,   $r19,   1
	fst.d	$f4,	$\AddrReg\(),	0
        addi.d  $\AddrReg\(),    $\AddrReg\(),    2*4
.else
	fst.d	$f\Vec1Reg\(),	$\AddrReg\(),	0
	addi.d	$\AddrReg\(),	$\AddrReg\(),	8
.endif
	vreplvei.w	$vr\Vec1Reg\(),	$vr\Vec1Reg\(),	2

        .endm

        .macro  OutputRow4Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
	vld	$vr4,	$\AddrReg\(),	0
	vreplvei.w      $vr1,  $vr0,   0
	vfmadd.s        $vr4,   $vr\Vec1Reg\(), $vr1,  $vr4
	vst     $vr4, $\AddrReg\(),    0
        addi.d  $\AddrReg\(),    $\AddrReg\(),    4*4
.else
	vst     $vr\Vec1Reg\(), $\AddrReg\(),    0
	addi.d	$\AddrReg\(),	$\AddrReg\(),	4*4
.endif
	vaddi.bu	$vr\Vec1Reg\(),	$vr\Vec2Reg\(),	0

        .endm

        .macro  OutputRow8Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
	vld     $vr4,   $\AddrReg\(),    0
        vld     $vr5,   $\AddrReg\(),    16
	vreplvei.w      $vr1,  $vr0,   0
	vfmadd.s        $vr4,   $vr\Vec1Reg\(), $vr1,  $vr4
        vfmadd.s        $vr5,   $vr\Vec2Reg\(), $vr1,  $vr5
	vst     $vr4, $\AddrReg\(),    0
        vst     $vr5, $\AddrReg\(),    16
        addi.d  $\AddrReg\(),    $\AddrReg\(),    8*4
.else
	vst     $vr\Vec1Reg\(), $\AddrReg\(),    0
	vst     $vr\Vec2Reg\(), $\AddrReg\(),    16
	addi.d  $\AddrReg\(),    $\AddrReg\(),    8*4
.endif
	vaddi.bu	$vr\Vec1Reg\(),	$vr\Vec3Reg\(),	0
	vaddi.bu	$vr\Vec2Reg\(), $vr\Vec4Reg\(), 0

        .endm

        .macro  OutputRow16Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
	vld     $vr4,   $\AddrReg\(),    0
        vld     $vr5,   $\AddrReg\(),    16
	vld     $vr6,   $\AddrReg\(),    8*4
	vld	$vr7,	$\AddrReg\(),	8*4+16
	vreplvei.w      $vr1,  $vr0,   0
	vfmadd.s        $vr4,   $vr\Vec1Reg\(), $vr1,  $vr4
	vfmadd.s        $vr5,   $vr\Vec2Reg\(), $vr1,  $vr5
	vfmadd.s        $vr6,   $vr\Vec3Reg\(), $vr1,  $vr6
	vfmadd.s        $vr7,   $vr\Vec4Reg\(), $vr1,  $vr7
	vst     $vr4,	$\AddrReg\(),    0
	vst	$vr5,	$\AddrReg\(),	16
	addi.d	$\AddrReg\(),	$\AddrReg\(),	16*4
	vst	$vr6,	$\AddrReg\(),	-8*4
	vst	$vr7,	$\AddrReg\(),	-8*4+16
.else
	vst     $vr\Vec1Reg\(),	$\AddrReg\(),    0
	vst	$vr\Vec2Reg\(),	$\AddrReg\(),	16
	addi.d	$\AddrReg\(),	$\AddrReg\(),	16*4
	vst	$vr\Vec3Reg\(),	$\AddrReg\(),	-8*4
	vst	$vr\Vec4Reg\(),	$\AddrReg\(),	-8*4+16
.endif

        .endm

//
// OutputBlock
//
// Generates the code to store the output block.
//

        .macro  OutputBlock Mode, Columns, Rows

	OutputRow\Columns\()Element \Mode\(),r6,16,17,18,19
.if \Rows\() >= 2
	OutputRow\Columns\()Element \Mode\(),r16,20,21,22,23
.endif
.if \Rows\() >= 4
	OutputRow\Columns\()Element \Mode\(),r17,24,25,26,27
	OutputRow\Columns\()Element \Mode\(),r18,28,29,30,31
.endif

        .endm

//
// ProcessRows
//
// Generates the code to process a compute and store the output block for a
// fixed number of rows.
//

        .macro  ProcessRows Mode, Rows

	addi.d	$r8,	$zero,	\Rows\()
	addi.d	$r19,	$zero,	8
	bge	$r19,	$r9,	.L\Mode\().ProcessRemainingCountN\Rows\()


.L\Mode\().ProcessNextColumnLoop16x\Rows\():
        ComputeBlockLoop \Mode\(),16,\Rows\()
.ifeqs "\Mode\()","Zero"
        MultiplyAlphaBlock 16,\Rows\()
.endif
	addi.d	$r9,	$r9,	-16
	srli.d  $r19,    $r9,    63
	bnez    $r19,	.L\Mode\().OutputMasked16x\Rows\().Block
        OutputBlock \Mode\(),16,\Rows\()
	add.d	$r4,	$r20,	$zero
	addi.d  $r19,    $zero,    8
	blt	$r19,	$r9,	.L\Mode\().ProcessNextColumnLoop16x\Rows\()
	beq	$r9,	$zero,	.L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN\Rows\():
        ComputeBlockLoop \Mode\(),8,\Rows\()
.ifeqs "\Mode\()","Zero"
        MultiplyAlphaBlock 8,\Rows\()
.endif

.L\Mode\().OutputMasked16x\Rows\().Block:
	andi	$r19,	$r9,	0x8
	beqz	$r19,	.L\Mode\().OutputRemaining7x\Rows\().Block
        OutputBlock \Mode\(),8,\Rows\()

.L\Mode\().OutputRemaining7x\Rows\().Block:
        andi     $r19,    $r9,    0x4
        beqz    $r19,	.L\Mode\().OutputRemaining3x\Rows\().Block
        OutputBlock \Mode\(),4,\Rows\()

.L\Mode\().OutputRemaining3x\Rows\().Block:
        andi     $r19,    $r9,    0x2
        beqz    $r19,	.L\Mode\().OutputRemaining1x\Rows\().Block
        OutputBlock \Mode\(),2,\Rows\()

.L\Mode\().OutputRemaining1x\Rows\().Block:
	andi     $r19,    $r9,    0x1
	beqz    $r19,	.L\Mode\().ExitKernel
        OutputBlock \Mode\(),1,\Rows\()

        .endm

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (x0) - Supplies the address of matrix A.

    B (x1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C (x2) - Supplies the address of matrix C.

    CountK (x3) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM (x4) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (x5) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda (x6) - Supplies the first dimension of matrix A.

    ldc (x7) - Supplies the first dimension of matrix C.

    Alpha (s0) - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/

        .macro  SgemmKernelNeonFunction Mode

        FUNCTION_ENTRY MlasSgemmKernel\Mode\()
        #save f24-f31
	fst.d $f24,   $sp,     -8
	fst.d $f25,   $sp,     -16
	fst.d $f26,   $sp,     -24
	fst.d $f27,   $sp,     -32
	fst.d $f28,   $sp,     -40
	fst.d $f29,   $sp,     -48
	fst.d $f30,   $sp,     -56
	fst.d $f31,   $sp,     -64
	addi.d        $sp,   $sp,   -64

	alsl.d	$r16,	$r11,	$r6,	2
	alsl.d	$r17,	$r11,	$r16,	2
	alsl.d	$r18,	$r11,	$r17,	2
	add.d	$r20,	$r4,	$zero

//
// Process 4 rows of the matrices.
//

	addi.d	$r19,	$zero,	4
	blt	$r8,	$r19,	.L\Mode\().ProcessCountMLessThan4
        ProcessRows \Mode\(),4

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
	add.d	$r4,	$r8,	$zero

        #pop f24-f31
	fld.d	$f31,	$sp,	0
	fld.d	$f30,	$sp,	8
	fld.d	$f29,	$sp,	16
	fld.d	$f28,	$sp,	24
	fld.d	$f27,	$sp,	32
	fld.d	$f26,	$sp,	40
	fld.d	$f25,	$sp,	48
	fld.d	$f24,	$sp,	56
	addi.d	$sp,	$sp,	64

	jirl	$r0,	$r1,	0

//
// Process 2 rows of the matrices.
//

.L\Mode\().ProcessCountMLessThan4:
	addi.d  $r19,    $zero,    2
	blt	$r8,	$r19,	.L\Mode\().ProcessCountMLessThan2
        ProcessRows \Mode\(),2
        b       .L\Mode\().ExitKernel

//
// Process 1 row of the matrices.
//

.L\Mode\().ProcessCountMLessThan2:
        ProcessRows \Mode\(),1
        b       .L\Mode\().ExitKernel

        .endm

        SgemmKernelNeonFunction Zero
        SgemmKernelNeonFunction Add

        .end
