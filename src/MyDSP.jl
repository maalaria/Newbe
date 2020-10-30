module MyDSP

export resample, bandpass, savitsky_golay, plot_spectrogram

using LinearAlgebra
using Plots
using Plots.PlotMeasures
using DSP

using ..MyUtils
#using ERec



# functions

function plot_spectrogram(s, fs, plt, sp)
    s = flatten(s)
    S = spectrogram(s, Int(25e-3*fs), Int(10e-3*fs); window=hanning)
    t = time(S)
    f = freq(S)
    #

    xtick_id = collect(1:100:size(S.power, 2))
    xtick_label = round.(Int.(collect(t))[xtick_id] ./ fs .*1000, digits=1)

    ytick_id = collect(1:50:size(S.power, 1))
    ytick_label = round.(collect(f)[ytick_id].*fs, digits=1)

    pl = heatmap!(log.(power(S)),
        xticks=( xtick_id, xtick_label ),
        yticks=( ytick_id, ytick_label ),
        grid=false,
        left_margin=50px,
        xlabel="ms",
        ylabel="Hz",
        subplot=sp
    );

    return pl
end




function savitsky_golay(x::Vector, windowSize::Integer, polyOrder::Integer; dv::Integer=0)
    #Polynomial smoothing with the Savitsky Golay filters
    #
    # Sources
    # ---------
    # Theory: http://www.ece.rutgers.edu/~orfanidi/intro2sp/orfanidis-i2sp.pdf
    # Python Example: http://wiki.scipy.org/Cookbook/SavitzkyGolay

    #Some error checking
    @assert isodd(windowSize) "Window size must be an odd integer."
    @assert polyOrder < windowSize "Polynomial order must me less than window size."
    halfWindow = Int((windowSize-1)/2)
    #Setup the S matrix of basis vectors.
    S = zeros(windowSize, polyOrder+1)
    for ct = 0:polyOrder
    	S[:,ct+1] = collect(-halfWindow:halfWindow).^(ct)
    end

    #Compute the filter coefficients for all orders
    #From the scipy code it seems pinv(S) and taking rows should be enough
    G = S*pinv(S'*S)

    #Slice out the derivative order we want
    filterCoeffs = G[:,dv+1] * factorial(dv);

    #Pad the signal with the endpoints and convolve with filter
    paddedX = [x[1]*ones(halfWindow), x, x[end]*ones(halfWindow)]
    y = conv(filterCoeffs[end:-1:1],  collect(Iterators.flatten(paddedX)))

    #Return the valid midsection
    return y[2*halfWindow+1:end-2*halfWindow]
end

end
