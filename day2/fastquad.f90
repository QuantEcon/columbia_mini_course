PURE FUNCTION QUAD(X0, N)
 IMPLICIT NONE
 INTEGER, PARAMETER :: DP=KIND(0.d0)                           
 REAL(dp), INTENT(IN) :: X0
 REAL(dp) :: QUAD
 INTEGER :: I
 INTEGER, INTENT(IN) :: N
 QUAD = X0
 DO I = 1, N - 1                                                
  QUAD = 4.0_dp * QUAD * real(1.0_dp - QUAD, dp)
 END DO
 RETURN
END FUNCTION QUAD

PROGRAM MAIN
 IMPLICIT NONE
 INTEGER, PARAMETER :: DP=KIND(0.d0)                          
 REAL(dp) :: START, FINISH, X, QUAD
 INTEGER :: N
 N = 10000000
 X = QUAD(0.2_dp, 10)
 CALL CPU_TIME(START)
 X = QUAD(0.2_dp, N)
 CALL CPU_TIME(FINISH)
 PRINT *,'last val = ', X
 PRINT *,'elapsed time = ', FINISH-START
END PROGRAM MAIN

