FT-LIS (private)
========

Falut-Tolerant LIS

Work harder for PPoPP 09/01/2013

Conjugate Gradient(CG)
-------- 
         ---------------------------------
         
         r(0)    = b - Ax(0)
 
         rho(-1) = 1
 
         p(0)    = (0,...,0)^T
         
         ----------------------------------
 
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
        ------------------------------------ 

1. vector relationship:

        p = z + beta * p
        
        q = A * p
        
        x = x + alpha * p
        
        r = r - alpha * q

2. checksum relationship map:

        checksum X <------ checksum P <------ checksum Z
                                |
                                ↓
        checksum R <------ checksum Q

3. check checksum X only, ideally

        Assume not error in proconditioning z=Br

        if cksQ* is incorrect -> cksR* is incorrect -> cksZ is incorrect(next iteration)
        
        if cksZ* is incorrect -> cksP* is incorrect -> cksX is incorrect

4. check checksum of X and R, practicaly

		step 3 is insufficient to detect all the error in all vectors

		add checking cksR
