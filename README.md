FT-LIS 
========

**Fault-Tolerant Iterative Solver for Linear Systems**

**Assume: no error during preconditioning**

Conjugate Gradient(CG)
-------- 
         ---------------------------------
         r(0)    = b - Ax(0) 
         rho(-1) = 1 
         p(0)    = (0,...,0)^T         
         ---------------------------------- 
         for k=1,2,... 
           z(k-1)    = M^(-1) * r(k-1)   
           rho(k-1)  = <r(k-1),z(k-1)>   
           beta      = rho(k-1) / rho(k-2)   
           p(k)      = z(k-1) + beta*p(k-1)   
           q(k)      = A * p(k)   
           dot_pq    = <p(k),q(k)>   
           alpha     = rho(k-1) / dot_pq   
           x(k)      = x(k-1) + alpha*p(k)   
           r(k)      = r(k-1) - alpha*q(k)
        ------------------------------------ 

1. vector relationship:

        z = M^(-1) * r
        p = z + beta * p        
        q = A * p        
        x = x + alpha * p        
        r = r - alpha * q

2. checksum relationship map:

        checksum X *--- checksum P *--- checksum Z
                            |
                            *
        checksum R *--- checksum Q

3. check checksum X only, ideally

        cksQ incorrect -> cksR incorrect -> cksZ incorrect(next iteration)     
        cksZ incorrect -> cksP incorrect -> cksX incorrect

4. check checksum of X and R, practically

        Assume no errors in preconditioning z = Br
		step 3 is insufficient to detect errors in all vectors
		so, add checking cksR


BiConjugate Gradient (BiCG)
--------
		 ----------------------------------------
		 r(0)    = b - Ax(0)
		 rtld(0) = r(0) or random
		 rho(-1) = 1
		 p(0)    = (0,...,0)^T
		 ptld(0) = (0,...,0)^T
		 ----------------------------------------
		 for k=1,2,...
		   z(k-1)    = M^-1 * r(k-1)
		   ztld(k-1) = M^-T * rtld(k-1)
		   rho(k-1)  = <z(k-1),rtld(k-1)>
		   beta      = rho(k-1) / rho(k-2)
		   p(k)      = z(k-1) + beta*p(k-1)
		   ptld(k)   = ztld(k-1) + beta*ptld(k-1)
		   q(k)      = A * p(k)
		   qtld(k)   = A^T * ptld(k)
		   tmpdot1   = <ptld(k),q(k)>
		   alpha     = rho(k-1) / tmpdot1
		   x(k)      = x(k-1) + alpha*p(k)
		   r(k)      = r(k-1) - alpha*q(k)
		   rtld(k)   = rtld(k-1) - alpha*qtld(k)
		 ----------------------------------------
1. vector relationship:

        z    = M^(-1) * r
        ztld = M^(-1) * rtld
        p    = z + beta * p
        ptld = ztld + beta * ptld
        q    = A * p
        qtld = At * ptld
        x    = x + aplha * p
        r    = r - aplha * q
        rtld = rtld - alpha * qtld
        
2. checksum relationship map:

        1. checksum X *--- checksum P *--- checksum Z
                               |
                               *
           checksum R *--- checksum Q
        
        2. checksum Rt *--- checksum Qt *--- checksum Pt *--- checksum Zt 
          
3. check checksum of X, R, and Rt         

        1. cksQ incorrect -> cksR incorrect -> cksZ incorrect(next iteration)     
           cksZ incorrect -> cksP incorrect -> cksX incorrect
        2. cksZt incorrect -> cksPt incorrect -> cksQt incorrect -> cksRt incorrect
     
Conjugate Gradient Squared (CGS)
--------
     -----------------------------------    
     r(0)    = b - Ax(0)
     rtld(0) = r(0) or random
     rho(-1) = 1
     p(0)    = (0,...,0)^T
     q(0)    = (0,...,0)^T
     -----------------------------------
     for k=1,2,...
       rho(k-1)  = <rtld,r(k-1)> 
       beta      = rho(k-1) / rho(k-2)  
       u(k)      = r(k-1) + beta*q(k-1) 
       p(k)      = u(k) + beta*(q(k-1) + beta*p(k-1)) 
       phat(k)   = M^-1 * p(k) 
       vhat(k)   = A * phat(k) 
       tmpdot1   = <rtld,vhat(k-1)> 
       alpha     = rho(k-1) / tmpdot1 
       q(k)      = u(k) - alpha*vhat(k) 
       phat(k)   = u(k) + q(k)
       uhat(k)   = M^-1 * (u(k) + q(k))
       x(k)      = x(k-1) + alpha*uhat(k)
       qhat(k)   = A * uhat(k)
       r(k)      = r(k-1) - alpha*qhat(k)
     ------------------------------------
     
1. vector relationship
        
        u    = r + beta * q
        p    = u + beta * (q + beta * p)
        phat = M^(-1) * p
        vhat = A * phat
        q    = u - alpha * vhat
        uhat = M^(-1) * (u + q)
        x    = x + alpha * uhat
        qhat = A * uhat
        r    = r - alpha * qhat
        
2. checksum relationship map

        cksU *--- cksR *--- cksQhat *--- cksUhat ---* cksX
         * |
         | *
        cksQ  ---*   cksP *--- cksU
          *            |
          |            *
        cksVhat *--- cksPhat
        
3. check checksum of P and X

        1. cksX incorrect
        2. cksQhat incorrect -> cksR incorrect -> cksU incorrect <- cksQ incorrect 
           cksU incorrect -> cksQ incorrect <- cksVhat incorrect
           cksU incorrect -> cksP incorrect <- cksQ incorrect
