class Solution ( object ) :
    def numberOfArithmeticSlices ( self , A ) :
        def numberOfArithmeticSlices2 ( self , A ) :
            result = 0
            for i in xrange ( 2 , len ( A ) ) :
                if A [ i ] - A [ i - 1 ] == A [ i - 1 ] - A [ i - 2 ] :
                    result += i
            return result
    