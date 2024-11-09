function gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
    d1 = (log(underlying/strike) + (b+ivol^2/2)*ttm)/(ivol*sqrt(ttm))
    d2 = d1 - ivol*sqrt(ttm)

    if call
        return underlying * exp((b-rf)*ttm) * cdf(Normal(),d1) - strike*exp(-rf*ttm)*cdf(Normal(),d2)
    else
        return strike*exp(-rf*ttm)*cdf(Normal(),-d2) - underlying*exp((b-rf)*ttm)*cdf(Normal(),-d1)
    end
    return nothing
end

function aggRisk(values, by_columns)
    # Custom aggregation logic here
    grouped_data = groupby(values, by_columns)
    # Summing up or other risk calculations on grouped data
    return combine(grouped_data, :RiskColumn => sum => :AggregatedRisk)
end
