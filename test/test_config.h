#pragma once

class TestConfig {
public:
    inline static unsigned long get_prng_seed()
    {
        return instance.prng_seed;
    }

    constexpr static unsigned long default_prng_seed = 42;

private:
    TestConfig();

    static TestConfig instance;
    unsigned long prng_seed;
};
