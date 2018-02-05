# Pandas Tips

### Stack or Glue DFs?
1. Make many smaller DataFrames and concatenate at the end (vs. continuously appending)

### When to use `object` type?
2. There's no integer NA (at the moment anyway), so if you have any missing values, represented by NaN, your otherwise integer column will be floats.
3. There's also no date dtype (distinct from datetime).
4. Consider the needs of your application: can you treat an integer 1 as 1.0? 
5. The last case of object dtype data is text data. Pandas doesn't have any fixed-width string dtypes, so you're stuck with python objects. 
6. There is an important exception here, and that's low-cardinality text data, for which you'll want to use the `category` dtype.

### Use Panda's built-in functions:

1. For largest values: `.nlargest(5)`
2. For smallest values: `.nsmallest(5).sort_values()`


### Use Vectors not `apply`
Instead of using a custom function `gcd_py` below:
```
pd.Series([gcd_py(*x) for x in pairs.itertuples(index=False)],
          index=pairs.index) 
```

```
r = pairs.apply(lambda x: gcd_py(x['LATITUDE_1'], x['LONGITUDE_1'],
                                 x['LATITUDE_2'], x['LONGITUDE_2']), axis=1);
```

```
r = gcd_vec(pairs['LATITUDE_1'], pairs['LONGITUDE_1'],
            pairs['LATITUDE_2'], pairs['LONGITUDE_2'])
```

### Pandas uses Cython optimized mean/std functions:

```
import random

def create_frame(n, n_groups):
    # just setup code, not benchmarking this
    stamps = pd.date_range('20010101', periods=n, freq='ms')
    random.shuffle(stamps.values)    
    return pd.DataFrame({'name': np.random.randint(0,n_groups,size=n),
                         'stamp': stamps,
                         'value': np.random.randint(0,n,size=n),
                         'value2': np.random.randn(n)})


df = create_frame(1000000,10000)

def f_apply(df):
    # Typical transform
    return df.groupby('name').value2.apply(lambda x: (x-x.mean())/x.std())

def f_unwrap(df):
    # "unwrapped"
    g = df.groupby('name').value2
    v = df.value2
    return (v-g.transform(np.mean))/g.transform(np.std)
```

andas GroupBy objects intercept calls for common functions like mean, sum, etc. and substitutes them with optimized Cython versions. So the unwrapped `.transform(np.mean)` and `.transform(np.std)` are fast, while the `x.mean` and `x.std` in the `.apply(lambda x: x - x.mean()/x.std())` aren't.




