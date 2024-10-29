#pragma once

namespace mlccpp{
    namespace concepts{

        template<typename T>
        concept SupportsLessThan = requires (T x) { x < x; };

        template<typename T>
        concept SupportsGreaterThan = requires (T x) { x > x; };

        template<typename T>
        concept SupportsLessOrEqualThan = requires (T x) { x <= x; };

        template<typename T>
        concept SupportsGreaterOrEqualThan = requires (T x) { x >= x; };
        
        template<typename T>
        concept SupportsEqual = requires (T x) { x == x; };

        template<typename T>
        concept SupportsNotEqual = requires (T x) { x != x; };

        template<typename T>
        concept SupportsTotalOrdering = SupportsLessThan<T> && 
            SupportsGreaterThan<T> && 
            SupportsLessOrEqualThan<T> && 
            SupportsGreaterOrEqualThan<T> && 
            SupportsEqual<T> &&
            SupportsNotEqual<T>;


        template<typename T>
        concept SupportsAddition = requires (T x) { x + x; };

        template<typename T>
        concept SupportsMultiplication = requires (T x) { x * x; };

        template<typename T>
        concept SupportsSubscription = requires (T x) { x - x; };

        template<typename T>
        concept SupportsDivision = requires (T x) { x / x; };

        template<typename T>
        concept SupportsArithmeticOperations = SupportsAddition<T> &&
            SupportsMultiplication<T> &&
            SupportsSubscription<T> &&
            SupportsDivision<T>;

    }
}