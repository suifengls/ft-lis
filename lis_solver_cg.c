/* Copyright (C) The Scalable Software Infrastructure Project. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   3. Neither the name of the project nor the names of its contributors 
      may be used to endorse or promote products derived from this software 
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE SCALABLE SOFTWARE INFRASTRUCTURE PROJECT
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE SCALABLE SOFTWARE INFRASTRUCTURE
   PROJECT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
   OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
	#include "lis_config.h"
#else
#ifdef HAVE_CONFIG_WIN_H
	#include "lis_config_win.h"
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_MALLOC_H
        #include <malloc.h>
#endif
#include <string.h>
#include <stdarg.h>
#ifdef USE_SSE2
	#include <emmintrin.h>
#endif
#ifdef _OPENMP
	#include <omp.h>
#endif
#ifdef USE_MPI
	#include <mpi.h>
#endif
#include "lislib.h"

/***************************************
 * Preconditioned Conjugate Gradient   *
 ***************************************
 r(0)    = b - Ax(0)
 rho(-1) = 1
 p(0)    = (0,...,0)^T
 ***************************************
 for k=1,2,...
   z(k-1)    = M^-1 * r(k-1)
   rho(k-1)  = <r(k-1),z(k-1)>
   beta      = rho(k-1) / rho(k-2)
   p(k)      = z(k-1) + beta*p(k-1)
   q(k)      = A * p(k)
   dot_pq    = <p(k),q(k)>
   alpha     = rho(k-1) / dot_pq
   x(k)      = x(k-1) + alpha*p(k)
   r(k)      = r(k-1) - alpha*q(k)
 ***************************************/

#define NWORK				4
#undef __FUNC__
#define __FUNC__ "lis_cg_check_params"
LIS_INT lis_cg_check_params(LIS_SOLVER solver)
{
	LIS_DEBUG_FUNC_IN;
	LIS_DEBUG_FUNC_OUT;
	return LIS_SUCCESS;
}

#undef __FUNC__
#define __FUNC__ "lis_cg_malloc_work"
LIS_INT lis_cg_malloc_work(LIS_SOLVER solver)
{
	LIS_VECTOR	*work;
	LIS_INT			i,j,worklen,err;

	LIS_DEBUG_FUNC_IN;

	// suifengls: extra work vectors 2 = Ones + sumA, 3 = ckpR, ckpP, ckpX
	worklen = NWORK + 2 + 3;
	work    = (LIS_VECTOR *)lis_malloc( worklen*sizeof(LIS_VECTOR),"lis_cg_malloc_work::work" );
	if( work==NULL )
	{
		LIS_SETERR_MEM(worklen*sizeof(LIS_VECTOR));
		return LIS_ERR_OUT_OF_MEMORY;
	}
	if( solver->precision==LIS_PRECISION_DEFAULT )
	{
		for(i=0;i<worklen;i++)
		{
			err = lis_vector_duplicate(solver->A,&work[i]);
			if( err ) break;
		}
	}
	else
	{
		for(i=0;i<worklen;i++)
		{
			err = lis_vector_duplicateex(LIS_PRECISION_QUAD,solver->A,&work[i]);
			if( err ) break;
		}
	}
	if( i<worklen )
	{
		for(j=0;j<i;j++) lis_vector_destroy(work[j]);
		lis_free(work);
		return err;
	}
	solver->worklen = worklen;
	solver->work    = work;

	LIS_DEBUG_FUNC_OUT;
	return LIS_SUCCESS;
}

#undef __FUNC__
#define __FUNC__ "lis_cg"
LIS_INT lis_cg(LIS_SOLVER solver)
{
	LIS_MATRIX A;
	LIS_PRECON M;
	LIS_VECTOR b,x;
	LIS_VECTOR r,z,p,q;
	LIS_SCALAR alpha, beta, rho, rho_old;
	LIS_SCALAR dot_pq;
	LIS_REAL   bnrm2, nrm2, tol;
	LIS_INT iter,maxiter,n,output,conv;
	double times,ptimes;

	// suifengls: ft-defined variables
	int rank; // tmp
	const LIS_INT CHECK_ITER = 15; // checking iteration
	const LIS_INT ERROR_ITER = 38; // introduce error iteration, set to 0 = no error introduced
	const LIS_INT CHKPT_ITER = 15; // checkpoint iteration
	const LIS_SCALAR eps = 1e-10;
	LIS_INT flag = 0; // 1 - error detected, 0 - no error
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // tmp
	LIS_SCALAR rerrX, rerrR;
	LIS_INT locN, gloN;
	LIS_VECTOR sumA, Ones;
	LIS_SCALAR cksA, cksR, cksZ, cksP, cksQ, cksX, checksum;
	// suifengls: checkpointing variables
	LIS_VECTOR ckpR, ckpP, ckpX;
	LIS_SCALAR ckprho = 0.0, ckpr, ckpp, ckpx;
	LIS_INT ckpiter = 0;
	LIS_INT num_error = 1;

	LIS_DEBUG_FUNC_IN;

	A       = solver->A;
	M       = solver->precon;
	b       = solver->b;
	x       = solver->x;
	n       = A->n;
	maxiter = solver->options[LIS_OPTIONS_MAXITER];
	output  = solver->options[LIS_OPTIONS_OUTPUT];
	conv    = solver->options[LIS_OPTIONS_CONV_COND];
	ptimes  = 0.0;


	z       = solver->work[0];
	q       = solver->work[1];
	r       = solver->work[2];
	p       = solver->work[3];
	rho_old = (LIS_SCALAR)1.0;
	beta    = (LIS_SCALAR)0.0;

	// suifengls: allocate extra vectors
	sumA	= solver->work[4];
	Ones	= solver->work[5];
	// suifengls: allocate checkpoint vectors
	ckpR	= solver->work[6];
	ckpP	= solver->work[7];
	ckpX	= solver->work[8];
	// get the global size of matrix A
	lis_matrix_get_size(A, &locN, &gloN);

	if( lis_solver_get_initial_residual(solver,NULL,NULL,r,&bnrm2) )
	{
		LIS_DEBUG_FUNC_OUT;
		return LIS_SUCCESS;
	}
	tol     = solver->tol;

	lis_vector_set_all(0.0,p);

	// suifengls: set ONE vector all to 1
	lis_vector_set_all(1.0,Ones);
	// A = A' -> get column sum of A, cksA = ones*A
	lis_matvec(A, Ones, sumA);
	// all sum of A -> cksA
	lis_vector_dot(Ones, sumA, &cksA);
	// sumA = sumA - a * cksA
	lis_vector_axpy(-cksA/(gloN+1), Ones, sumA);
	// cksA = sum of sumA
	lis_vector_dot(Ones, sumA, &cksA);
	//printf("AN = %d, cksA = %e\n", gloN, cksA);
	lis_output_vector(sumA, LIS_FMT_MM, "sumA");
	// initialize all checksum
	cksR = 0.0, cksZ = 0.0, cksP = 0.0, cksQ = 0.0, cksX = 0.0, checksum = 0.0;
	lis_vector_dot(Ones, r, &cksR);
	lis_vector_dot(Ones, x, &cksX);
	
	flag = 0;  
	num_error = 2;
	for( iter=1; iter<=maxiter; iter++ )
	{
		/* z = M^-1 * r */
		times = lis_wtime();
		lis_psolve(solver,r,z);
		ptimes += lis_wtime() - times;

		// assuming no error in precondition
		// suifengls: checksum Z
		lis_vector_dot(Ones, z, &cksZ);

		/* rho = <r,z> */
		lis_vector_dot(r,z,&rho);

		/* beta = rho / rho_old */
		beta = rho / rho_old;

		/* p = z + beta*p       */
		lis_vector_xpay(z,beta,p);
		// suifengls: checksum P
		cksP = cksZ + beta * cksP;
		
		/* q = Ap */
		LIS_MATVEC(A,p,q);

		// suifengls: checksum Q
		lis_vector_dot(p, sumA, &cksQ);
		cksQ = cksQ + cksA * cksP;
		
		// suifengls: introduce an error
		if(!rank && iter == ERROR_ITER)
		//if(num_error > 0 && !rank && iter == ERROR_ITER)
		{
			printf("========== Introducing an error at iteration %d ==========\n", iter);
			lis_vector_set_value(LIS_INS_VALUE, 0, 16, q); 
			num_error--;
		}
		/* dot_pq = <p,q> */
		lis_vector_dot(p,q,&dot_pq);

		/* breakdown check */
		if( dot_pq==0.0 )
		{
			solver->retcode   = LIS_BREAKDOWN;
			solver->iter      = iter;
			solver->resid     = nrm2;
			LIS_DEBUG_FUNC_OUT;
			return LIS_BREAKDOWN;
		}
		
		/* alpha = rho / dot_pq */
		alpha = rho / dot_pq;
		
		/* x = x + alpha*p */
		lis_vector_axpy(alpha,p,x);
		// suifengls: checksum X
		cksX = cksX + alpha * cksP;
		
		/* r = r - alpha*q */
		lis_vector_axpy(-alpha,q,r);
		// suifengls: checksum R
		cksR = cksR - alpha * cksQ;
		
		rho_old = rho;
		//suifengls: checking, checkpointing, recovering
		if(iter % CHECK_ITER == 0) 
		{
			// suifengls: checking cksX
			lis_vector_dot(Ones, x, &checksum);
			rerrX = fabs(checksum - cksX)/fabs(cksX);
			if(rerrX > eps && !flag)
			{
				flag = 1; // error detected!
				if(!rank)
					printf("========== error detected in X: %e at iteration %d ==========\n", rerrX, iter);
				//printf("sum of X = %e, checksum = %e\n", checksum, cksX);
			}

			// suifengls: checking cksR
			lis_vector_dot(Ones, r, &checksum);
			rerrR = fabs(checksum - cksR)/fabs(cksR);
			if(iter % CHECK_ITER == 0 && (fabs(cksR) > eps && fabs(checksum) > eps) && rerrR > eps && !flag)
			{
				flag = 1; // error detected!
				if(!rank)
					printf("========== error detected in R: %e at iteration %d ==========\n", rerrR, iter);
				//printf("sum of R = %e, checksum = %e\n", checksum, cksR);
			}

			// suifengls: checkpointing and recovery
			if(!flag) // no error, make a checkpointing
			{
				if(!rank)
					printf("========== Checkpointing at itertion %d ==========\n", iter);
				ckpiter = iter;
				ckprho = rho_old;
				lis_vector_copy(r, ckpR);
				ckpr = cksR;
				lis_vector_copy(p, ckpP);
				ckpp = cksP;
				lis_vector_copy(x, ckpX);
				ckpx = cksX;
			}
			else  // error detected, recovery!
			{
				if(!rank)
					printf("========== Rollback to itertion %d ==========\n", ckpiter);
				// rollback
				//iter = ckpiter;
				rho_old = ckprho;
				lis_vector_copy(ckpR, r);
				cksR = ckpr;
				lis_vector_copy(ckpP, p);
				cksP = ckpp;
				lis_vector_copy(ckpX, x);
				cksX = ckpx;
				flag = 0;
			}
		}

		/* convergence check */
		lis_solver_get_residual[conv](r,solver,&nrm2);
		if( output )
		{
			if( output & LIS_PRINT_MEM ) solver->residual[iter] = nrm2;
			if( output & LIS_PRINT_OUT && A->my_rank==0 ) lis_print_rhistory(iter,nrm2);
		}
		
		if( tol >= nrm2 )
		{
			solver->retcode    = LIS_SUCCESS;
			solver->iter       = iter;
			solver->resid      = nrm2;
			solver->ptimes     = ptimes;
			LIS_DEBUG_FUNC_OUT;
			return LIS_SUCCESS;
		}
	}

	solver->retcode   = LIS_MAXITER;
	solver->iter      = iter;
	solver->resid     = nrm2;
	LIS_DEBUG_FUNC_OUT;
	return LIS_MAXITER;
}

#ifdef USE_QUAD_PRECISION
#undef __FUNC__
#define __FUNC__ "lis_cg_quad"
LIS_INT lis_cg_quad(LIS_SOLVER solver)
{
	LIS_MATRIX A;
	LIS_PRECON M;
	LIS_VECTOR b,x;
	LIS_VECTOR r,z,p,q;
	LIS_QUAD_PTR alpha, beta, rho, rho_old;
	LIS_QUAD_PTR dot_pq;
	LIS_REAL   bnrm2, nrm2, tol;
	LIS_INT iter,maxiter,n,output,conv;
	double times,ptimes;

	LIS_DEBUG_FUNC_IN;

	A       = solver->A;
	M       = solver->precon;
	b       = solver->b;
	x       = solver->x;
	n       = A->n;
	maxiter = solver->options[LIS_OPTIONS_MAXITER];
	output  = solver->options[LIS_OPTIONS_OUTPUT];
	conv    = solver->options[LIS_OPTIONS_CONV_COND];
	ptimes  = 0.0;

	LIS_QUAD_SCALAR_MALLOC(alpha,0,1);
	LIS_QUAD_SCALAR_MALLOC(beta,1,1);
	LIS_QUAD_SCALAR_MALLOC(rho,2,1);
	LIS_QUAD_SCALAR_MALLOC(rho_old,3,1);
	LIS_QUAD_SCALAR_MALLOC(dot_pq,4,1);
	
	z       = solver->work[0];
	q       = solver->work[1];
	r       = solver->work[2];
	p       = solver->work[3];
	rho_old.hi[0] = 1.0;
	rho_old.lo[0] = 0.0;
	beta.hi[0] = 0.0;
	beta.lo[0] = 0.0;

	if( lis_solver_get_initial_residual(solver,NULL,NULL,r,&bnrm2) )
	{
		LIS_DEBUG_FUNC_OUT;
		return LIS_SUCCESS;
	}
	tol     = solver->tol;

	lis_vector_set_allex_nm(0.0,p);
	
	for( iter=1; iter<=maxiter; iter++ )
	{
		/* z = M^-1 * r */
		times = lis_wtime();
		lis_psolve(solver,r,z);
		ptimes += lis_wtime() - times;

		/* rho = <r,z> */
		lis_vector_dotex_mmm(r,z,&rho);

		/* beta = rho / rho_old */
		lis_quad_div((LIS_QUAD *)beta.hi,(LIS_QUAD *)rho.hi,(LIS_QUAD *)rho_old.hi);

		/* p = z + beta*p       */
		lis_vector_xpayex_mmm(z,beta,p);

		/* q = Ap */
		LIS_MATVEC(A,p,q);
		
		/* dot_pq = <p,q> */
		lis_vector_dotex_mmm(p,q,&dot_pq);

		/* breakdown check */
		if( dot_pq.hi[0]==0.0 && dot_pq.lo[0]==0.0 )
		{
			solver->retcode   = LIS_BREAKDOWN;
			solver->iter      = iter;
			solver->resid     = nrm2;
			LIS_DEBUG_FUNC_OUT;
			return LIS_BREAKDOWN;
		}
		
		/* alpha = rho / dot_pq */
		lis_quad_div((LIS_QUAD *)alpha.hi,(LIS_QUAD *)rho.hi,(LIS_QUAD *)dot_pq.hi);
		
		/* x = x + alpha*p */
		lis_vector_axpyex_mmm(alpha,p,x);
		
		/* r = r - alpha*q */
		lis_quad_minus((LIS_QUAD *)alpha.hi);
		lis_vector_axpyex_mmm(alpha,q,r);

		/* convergence check */
		lis_solver_get_residual[conv](r,solver,&nrm2);
		if( output )
		{
			if( output & LIS_PRINT_MEM ) solver->residual[iter] = nrm2;
			if( output & LIS_PRINT_OUT && A->my_rank==0 ) lis_print_rhistory(iter,nrm2);
		}
		
		if( tol > nrm2 )
		{
			solver->retcode    = LIS_SUCCESS;
			solver->iter       = iter;
			solver->resid      = nrm2;
			solver->ptimes     = ptimes;
			LIS_DEBUG_FUNC_OUT;
			return LIS_SUCCESS;
		}
		rho_old.hi[0] = rho.hi[0];
		rho_old.lo[0] = rho.lo[0];
	}

	solver->retcode   = LIS_MAXITER;
	solver->iter      = iter;
	solver->resid     = nrm2;
	LIS_DEBUG_FUNC_OUT;
	return LIS_MAXITER;
}

#undef __FUNC__
#define __FUNC__ "lis_cg_switch"
LIS_INT lis_cg_switch(LIS_SOLVER solver)
{
	LIS_MATRIX A;
	LIS_PRECON M;
	LIS_VECTOR b,x;
	LIS_VECTOR r,z,p,q;
	LIS_QUAD_PTR alpha, beta, rho, rho_old;
	LIS_QUAD_PTR dot_pq;
	LIS_REAL   bnrm2, nrm2, tol, tol2;
	LIS_INT iter,maxiter,n,output,conv;
	LIS_INT iter2,maxiter2;
	double times,ptimes;

	LIS_DEBUG_FUNC_IN;

	A       = solver->A;
	M       = solver->precon;
	b       = solver->b;
	x       = solver->x;
	n       = A->n;
	maxiter  = solver->options[LIS_OPTIONS_MAXITER];
	maxiter2 = solver->options[LIS_OPTIONS_SWITCH_MAXITER];
	output   = solver->options[LIS_OPTIONS_OUTPUT];
	conv    = solver->options[LIS_OPTIONS_CONV_COND];
	tol      = solver->params[LIS_PARAMS_RESID-LIS_OPTIONS_LEN];
	tol2     = solver->params[LIS_PARAMS_SWITCH_RESID-LIS_OPTIONS_LEN];
	ptimes  = 0.0;

	LIS_QUAD_SCALAR_MALLOC(alpha,0,1);
	LIS_QUAD_SCALAR_MALLOC(beta,1,1);
	LIS_QUAD_SCALAR_MALLOC(rho,2,1);
	LIS_QUAD_SCALAR_MALLOC(rho_old,3,1);
	LIS_QUAD_SCALAR_MALLOC(dot_pq,4,1);
	
	z       = solver->work[0];
	q       = solver->work[1];
	r       = solver->work[2];
	p       = solver->work[3];
	rho_old.hi[0] = 1.0;
	rho_old.lo[0] = 0.0;
	beta.hi[0] = 0.0;
	beta.lo[0] = 0.0;

	if( lis_solver_get_initial_residual(solver,NULL,NULL,r,&bnrm2) )
	{
		LIS_DEBUG_FUNC_OUT;
		return LIS_SUCCESS;
	}
	tol2     = solver->tol_switch;

	lis_vector_set_allex_nm(0.0,p);
	r->precision = LIS_PRECISION_DEFAULT;
	p->precision = LIS_PRECISION_DEFAULT;
	
	for( iter=1; iter<=maxiter2; iter++ )
	{
		/* z = M^-1 * r */
		times = lis_wtime();
		lis_psolve(solver,r,z);
		ptimes += lis_wtime() - times;

		/* rho = <r,z> */
		lis_vector_dot(r,z,&rho.hi[0]);

		/* beta = rho / rho_old */
		beta.hi[0] = rho.hi[0] / rho_old.hi[0];

		/* p = z + beta*p       */
		lis_vector_xpay(z,beta.hi[0],p);
		
		/* q = Ap */
		LIS_MATVEC(A,p,q);
		
		/* dot_pq = <p,q> */
		lis_vector_dot(p,q,&dot_pq.hi[0]);

		/* breakdown check */
		if( dot_pq.hi[0]==0.0 )
		{
			solver->retcode   = LIS_BREAKDOWN;
			solver->iter      = iter;
			solver->iter2     = iter;
			solver->resid     = nrm2;
			LIS_DEBUG_FUNC_OUT;
			return LIS_BREAKDOWN;
		}
		
		/* alpha = rho / dot_pq */
		alpha.hi[0] = rho.hi[0] / dot_pq.hi[0];
		
		/* x = x + alpha*p */
		lis_vector_axpy(alpha.hi[0],p,x);
		
		/* r = r - alpha*q */
		lis_vector_axpy(-alpha.hi[0],q,r);

		/* convergence check */
		lis_solver_get_residual[conv](r,solver,&nrm2);
		if( output )
		{
			if( output & LIS_PRINT_MEM ) solver->residual[iter] = nrm2;
			if( output & LIS_PRINT_OUT && A->my_rank==0 ) lis_print_rhistory(iter,nrm2);
		}
		
		if( tol2 >= nrm2 )
		{
			solver->iter       = iter;
			solver->iter2     = iter;
			solver->ptimes     = ptimes;
			break;
		}
		rho_old.hi[0] = rho.hi[0];
	}

	r->precision = LIS_PRECISION_QUAD;
	p->precision = LIS_PRECISION_QUAD;
	
	solver->options[LIS_OPTIONS_INITGUESS_ZEROS] = LIS_FALSE;
	lis_vector_copyex_mn(x,solver->xx);
	rho_old.hi[0] = 1.0;
	beta.hi[0] = 0.0;
	lis_solver_get_initial_residual(solver,NULL,NULL,r,&bnrm2);
	tol     = solver->tol;

	lis_vector_set_allex_nm(0.0,p);
	
	for( iter2=iter+1; iter2<=maxiter; iter2++ )
	{
		/* z = M^-1 * r */
		times = lis_wtime();
		lis_psolve(solver,r,z);
		ptimes += lis_wtime() - times;

		/* rho = <r,z> */
		lis_vector_dotex_mmm(r,z,&rho);

		/* beta = rho / rho_old */
		lis_quad_div((LIS_QUAD *)beta.hi,(LIS_QUAD *)rho.hi,(LIS_QUAD *)rho_old.hi);

		/* p = z + beta*p       */
		lis_vector_xpayex_mmm(z,beta,p);

		/* q = Ap */
		LIS_MATVEC(A,p,q);
		
		/* dot_pq = <p,q> */
		lis_vector_dotex_mmm(p,q,&dot_pq);

		/* breakdown check */
		if( dot_pq.hi[0]==0.0 && dot_pq.lo[0]==0.0 )
		{
			solver->retcode   = LIS_BREAKDOWN;
			solver->iter      = iter2;
			solver->resid     = nrm2;
			LIS_DEBUG_FUNC_OUT;
			return LIS_BREAKDOWN;
		}
		
		/* alpha = rho / dot_pq */
		lis_quad_div((LIS_QUAD *)alpha.hi,(LIS_QUAD *)rho.hi,(LIS_QUAD *)dot_pq.hi);
		
		/* x = x + alpha*p */
		lis_vector_axpyex_mmm(alpha,p,x);
		
		/* r = r - alpha*q */
		lis_quad_minus((LIS_QUAD *)alpha.hi);
		lis_vector_axpyex_mmm(alpha,q,r);

		/* convergence check */
		lis_solver_get_residual[conv](r,solver,&nrm2);
		if( output )
		{
			if( output & LIS_PRINT_MEM ) solver->residual[iter2] = nrm2;
			if( output & LIS_PRINT_OUT && A->my_rank==0 ) lis_print_rhistory(iter,nrm2);
		}
		
		if( tol > nrm2 )
		{
			solver->retcode    = LIS_SUCCESS;
			solver->iter       = iter2;
			solver->resid      = nrm2;
			solver->ptimes     = ptimes;
			LIS_DEBUG_FUNC_OUT;
			return LIS_SUCCESS;
		}
		rho_old.hi[0] = rho.hi[0];
		rho_old.lo[0] = rho.lo[0];
	}

	solver->retcode   = LIS_MAXITER;
	solver->iter      = iter2;
	solver->resid     = nrm2;
	LIS_DEBUG_FUNC_OUT;
	return LIS_MAXITER;
}
#endif

/***************************************
 * Preconditioned Conjugate Residual   *
 ***************************************
 r(0)    = b - Ax(0)
 p(0)    = M^-1 * r(0)
 q(0)    = Ap(0)
 z(0)    = p(0)
 ***************************************
 for k=1,2,...
   qtld(k-1) = M^-1 * q(k-1)
   rho(k-1)  = <qtld(k-1),q(k-1)>
   dot_rq    = <r(k-1),qtld(k-1)>
   alpha     = dot_rq / rho(k-1)
   x(k)      = x(k-1) + alpha*p(k-1)
   r(k)      = r(k-1) - alpha*q(k-1)
   z(k)      = z(k-1) - alpha*qtld(k-1)
   az(k)     = A * z(k)
   dot_zq    = <az(k),qtld(k-1)>
   beta      =  -dot_zq / rho(k-1)
   p(k)      = z(k)  + beta*p(k-1)
   q(k)      = az(k) + beta*q(k-1)
 ***************************************/
#undef NWORK
#define NWORK				6
#undef __FUNC__
#define __FUNC__ "lis_cr_check_params"
LIS_INT lis_cr_check_params(LIS_SOLVER solver)
{
	LIS_DEBUG_FUNC_IN;
	LIS_DEBUG_FUNC_OUT;
	return LIS_SUCCESS;
}

#undef __FUNC__
#define __FUNC__ "lis_cr_malloc_work"
LIS_INT lis_cr_malloc_work(LIS_SOLVER solver)
{
	LIS_VECTOR	*work;
	LIS_INT			i,j,worklen,err;

	LIS_DEBUG_FUNC_IN;

	worklen = NWORK;
	work    = (LIS_VECTOR *)lis_malloc( worklen*sizeof(LIS_VECTOR),"lis_cr_malloc_work::work" );
	if( work==NULL )
	{
		LIS_SETERR_MEM(worklen*sizeof(LIS_VECTOR));
		return LIS_ERR_OUT_OF_MEMORY;
	}
	if( solver->precision==LIS_PRECISION_DEFAULT )
	{
		for(i=0;i<worklen;i++)
		{
			err = lis_vector_duplicate(solver->A,&work[i]);
			if( err ) break;
		}
	}
	else
	{
		for(i=0;i<worklen;i++)
		{
			err = lis_vector_duplicateex(LIS_PRECISION_QUAD,solver->A,&work[i]);
			if( err ) break;
		}
	}
	if( i<worklen )
	{
		for(j=0;j<i;j++) lis_vector_destroy(work[j]);
		lis_free(work);
		return err;
	}
	solver->worklen = worklen;
	solver->work    = work;

	LIS_DEBUG_FUNC_OUT;
	return LIS_SUCCESS;
}

#undef __FUNC__
#define __FUNC__ "lis_cr"
LIS_INT lis_cr(LIS_SOLVER solver)
{
	LIS_MATRIX A;
	LIS_PRECON M;
	LIS_VECTOR b,x;
	LIS_VECTOR r,z,p,q, qtld, az;
	LIS_SCALAR alpha, beta, rho;
	LIS_SCALAR dot_rq, dot_zq;
	LIS_REAL   bnrm2, nrm2, tol;
	LIS_INT iter,maxiter,n,output,conv;
	double times,ptimes;

	LIS_DEBUG_FUNC_IN;

	A       = solver->A;
	M       = solver->precon;
	b       = solver->b;
	x       = solver->x;
	n       = A->n;
	maxiter = solver->options[LIS_OPTIONS_MAXITER];
	output  = solver->options[LIS_OPTIONS_OUTPUT];
	conv    = solver->options[LIS_OPTIONS_CONV_COND];
	ptimes  = 0.0;


	z       = solver->work[0];
	q       = solver->work[1];
	r       = solver->work[2];
	p       = solver->work[3];
	qtld    = solver->work[4];
	az      = solver->work[5];


	if( lis_solver_get_initial_residual(solver,NULL,NULL,r,&bnrm2) )
	{
		LIS_DEBUG_FUNC_OUT;
		return LIS_SUCCESS;
	}
	tol     = solver->tol;

	times = lis_wtime();
	lis_psolve(solver,r,p);
	ptimes += lis_wtime() - times;
	LIS_MATVEC(A,p,q);
	lis_vector_copy(p,z);

	for( iter=1; iter<=maxiter; iter++ )
	{
		/* qtld = M^-1 * q */
		times = lis_wtime();
		lis_psolve(solver,q,qtld);
		ptimes += lis_wtime() - times;

		/* rho = <qtld,q> */
		lis_vector_dot(qtld,q,&rho);

		/* breakdown check */
		if( rho==0.0 )
		{
			solver->retcode   = LIS_BREAKDOWN;
			solver->iter      = iter;
			solver->resid     = nrm2;
			LIS_DEBUG_FUNC_OUT;
			return LIS_BREAKDOWN;
		}
		
		/* dot_rq = <r,qtld> */
		lis_vector_dot(r,qtld,&dot_rq);

		/* alpha = dot_rq / rho */
		alpha = dot_rq / rho;
		
		/* x = x + alpha*p */
		lis_vector_axpy(alpha,p,x);
		
		/* r = r - alpha*q */
		lis_vector_axpy(-alpha,q,r);

		/* convergence check */
		lis_solver_get_residual[conv](r,solver,&nrm2);
		if( output )
		{
			if( output & LIS_PRINT_MEM ) solver->residual[iter] = nrm2;
			if( output & LIS_PRINT_OUT && A->my_rank==0 ) lis_print_rhistory(iter,nrm2);
		}
		
		if( tol >= nrm2 )
		{
			solver->retcode    = LIS_SUCCESS;
			solver->iter       = iter;
			solver->resid      = nrm2;
			solver->ptimes     = ptimes;
			LIS_DEBUG_FUNC_OUT;
			return LIS_SUCCESS;
		}

		/* z = z - alpha*qtld       */
		lis_vector_axpy(-alpha,qtld,z);

		/* az = Az */
		LIS_MATVEC(A,z,az);

		/* dot_zq = <az,qtld> */
		lis_vector_dot(az,qtld,&dot_zq);

		/* beta = -dot_zq / rho */
		beta = -dot_zq / rho;

		/* p = z + beta*p       */
		lis_vector_xpay(z,beta,p);
		
		/* q = az + beta*q      */
		lis_vector_xpay(az,beta,q);
	}

	solver->retcode   = LIS_MAXITER;
	solver->iter      = iter;
	solver->resid     = nrm2;
	LIS_DEBUG_FUNC_OUT;
	return LIS_MAXITER;
}

#ifdef USE_QUAD_PRECISION
#undef __FUNC__
#define __FUNC__ "lis_cr_quad"
LIS_INT lis_cr_quad(LIS_SOLVER solver)
{
	LIS_MATRIX A;
	LIS_PRECON M;
	LIS_VECTOR b,x;
	LIS_VECTOR r,z,p,q, qtld, az;
	LIS_QUAD_PTR alpha, beta, rho;
	LIS_QUAD_PTR dot_rq, dot_zq;
	LIS_REAL   bnrm2, nrm2, tol;
	LIS_INT iter,maxiter,n,output,conv;
	double times,ptimes;

	LIS_DEBUG_FUNC_IN;

	A       = solver->A;
	M       = solver->precon;
	b       = solver->b;
	x       = solver->x;
	n       = A->n;
	maxiter = solver->options[LIS_OPTIONS_MAXITER];
	output  = solver->options[LIS_OPTIONS_OUTPUT];
	conv    = solver->options[LIS_OPTIONS_CONV_COND];
	ptimes  = 0.0;

	LIS_QUAD_SCALAR_MALLOC(alpha,0,1);
	LIS_QUAD_SCALAR_MALLOC(beta,1,1);
	LIS_QUAD_SCALAR_MALLOC(rho,2,1);
	LIS_QUAD_SCALAR_MALLOC(dot_rq,3,1);
	LIS_QUAD_SCALAR_MALLOC(dot_zq,4,1);

	z       = solver->work[0];
	q       = solver->work[1];
	r       = solver->work[2];
	p       = solver->work[3];
	qtld    = solver->work[4];
	az      = solver->work[5];


	if( lis_solver_get_initial_residual(solver,NULL,NULL,r,&bnrm2) )
	{
		LIS_DEBUG_FUNC_OUT;
		return LIS_SUCCESS;
	}
	tol     = solver->tol;

	times = lis_wtime();
	lis_psolve(solver,r,p);
	ptimes += lis_wtime() - times;
	LIS_MATVEC(A,p,q);
	lis_vector_copyex_mm(p,z);

	for( iter=1; iter<=maxiter; iter++ )
	{
		/* qtld = M^-1 * q */
		times = lis_wtime();
		lis_psolve(solver,q,qtld);
		ptimes += lis_wtime() - times;

		/* rho = <qtld,q> */
		lis_vector_dotex_mmm(qtld,q,&rho);

		/* breakdown check */
		if( rho.hi[0]==0.0 && rho.lo[0]==0.0 )
		{
			solver->retcode   = LIS_BREAKDOWN;
			solver->iter      = iter;
			solver->resid     = nrm2;
			LIS_DEBUG_FUNC_OUT;
			return LIS_BREAKDOWN;
		}
		
		/* dot_rq = <r,qtld> */
		lis_vector_dotex_mmm(r,qtld,&dot_rq);

		/* alpha = dot_rq / rho */
		lis_quad_div((LIS_QUAD *)alpha.hi,(LIS_QUAD *)dot_rq.hi,(LIS_QUAD *)rho.hi);
		
		/* x = x + alpha*p */
		lis_vector_axpyex_mmm(alpha,p,x);
		
		/* r = r - alpha*q */
		lis_quad_minus((LIS_QUAD *)alpha.hi);
		lis_vector_axpyex_mmm(alpha,q,r);

		/* convergence check */
		lis_solver_get_residual[conv](r,solver,&nrm2);
		if( output )
		{
			if( output & LIS_PRINT_MEM ) solver->residual[iter] = nrm2;
			if( output & LIS_PRINT_OUT && A->my_rank==0 ) lis_print_rhistory(iter,nrm2);
		}
		
		if( tol >= nrm2 )
		{
			solver->retcode    = LIS_SUCCESS;
			solver->iter       = iter;
			solver->resid      = nrm2;
			solver->ptimes     = ptimes;
			LIS_DEBUG_FUNC_OUT;
			return LIS_SUCCESS;
		}

		/* z = z - alpha*qtld       */
		lis_vector_axpyex_mmm(alpha,qtld,z);

		/* az = Az */
		LIS_MATVEC(A,z,az);

		/* dot_zq = <az,qtld> */
		lis_vector_dotex_mmm(az,qtld,&dot_zq);

		/* beta = -dot_zq / rho */
		lis_quad_div((LIS_QUAD *)beta.hi,(LIS_QUAD *)dot_zq.hi,(LIS_QUAD *)rho.hi);
		lis_quad_minus((LIS_QUAD *)beta.hi);

		/* p = z + beta*p       */
		lis_vector_xpayex_mmm(z,beta,p);
		
		/* q = az + beta*q      */
		lis_vector_xpayex_mmm(az,beta,q);
	}

	solver->retcode   = LIS_MAXITER;
	solver->iter      = iter;
	solver->resid     = nrm2;
	LIS_DEBUG_FUNC_OUT;
	return LIS_MAXITER;
}
#endif
