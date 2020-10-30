
"""
Data structure
--------------
mutable struct: SELF.recording

Functions
---------
SELF.recording.filter()

SELF.recording.resample()

SELF.recording.detect_spikes()
"""
module Neurons

export recording, normalize_trace, filter_trace, resample, detect_spikes, detect_spikes1, get_PSD, extract_spikes, extract_features, cluster_spikes, find_redundant, extract_isih,
		plot_detection, plot_waveforms

using DSP
using Images
using Statistics
using MultivariateStats
using GaussianMixtures
using Clustering
using LinearAlgebra
using Distances
using PyPlot

using ..MyUtils

# main type
"""
Fields
------
+ file::String
+ sf::Real
+ sf_preprocessed::Real
+ n_channels::Real
+ trace::AbstractArray # supposed to contain sample X channel
+ trial_marker::AbstractArray
+ trace_preprocessed::AbstractArray
+ preprocessing_steps::Vector{String}
+ spike_idx::AbstractArray
+ detection_threshold::AbstractArray
+ isih::AbstractArray
+ spikeforms::AbstractArray # channel (val_source) X channel (t_source) X time of spike/spikeform X sample
+ features::AbstractArray # channel X feature space representation of spikes
+ spike_class::AbstractArray # channel X clusters (as spike indices)
+ BIC::AbstractArray
"""
mutable struct recording
    ### FIELDS
    file::String
    sf::Real
    sf_preprocessed::Real
    n_channels::Real
    trace::Array{Array{Float64,2}}#AbstractArray # supposed to contain sample X channel
    trial_marker::AbstractArray
    trace_preprocessed::Array{Array{Float64,2}}#AbstractArray
    preprocessing_steps::Vector{String}
    spike_idx::AbstractArray
	detection_threshold::AbstractArray
    isih::AbstractArray
    spikeforms::AbstractArray # channel (val_source) X channel (t_source) X time of spike/spikeform X sample
    features::AbstractArray # channel X feature space representation of spikes
    spike_class::AbstractArray # channel X clusters (as spike indices)
    BIC::AbstractArray

    ### CONSTRUCTOR
    # recording(file, sf, trace, trial_marker) =
    #    new(file, sf, 0, trace, trial_marker, Array{Float64,2}(undef,0,0), Vector{String}(undef,0), Array{Float64,3}Array{Float64,2}(undef,0,0,0))
    # recording() = new("", 0, 0, Array{Float64,2}(undef,0,0), Array{Float64,2}(undef,0,0), Array{Float64,2}(undef,0,0), Vector{String}(undef,0), Array{Float64,3}(undef,0,0,0))
    recording() = new("", 0, 0, 0, [], [], [], Vector{String}(undef,0), [], [], [], [], [], [])
end



# functions

function normalize_trace(n::Neurons.recording, what="trace")


	x = deepcopy( getfield( n, Symbol(what) ) )
	sd = std.(x)
	mn = mean.(x)
	[x[ii]=(x[ii] .- mn[ii]) ./ sd[ii] for ii in 1:n.n_channels] ### normalization
	n.trace_preprocessed = x
	n.preprocessing_steps = ["normalized"];

end

"""
Documentation ...
"""
function filter_trace(n::Neurons.recording, filter_type::String, freqs::AbstractArray, filter_design::AbstractArray, filter_order::AbstractArray, bandwidth::AbstractArray, from_chnnls::AbstractArray, what::String="trace")
"""
    filter(n::recording, band::Tuple{Real}, filter_type)
    filter_type: lowpass, highpass, bandpass, notch
    freqs = highpass: freq | lowpass: freq | passband (Wn1,Wn2) frequency | Notch: array of frequencies | CommonAverage: channels used for averaging
    what: spcifies whether filter works on trace or trace_preprocessed. If the latter trace_preprocessed is replaced, respectively.
"""

    ######################
    ### raw trace
    ######################
    if what == "trace"
        nyq = n.sf/2;
        freqs_n = freqs./nyq; # normalized to nquist frequncy
        n.trace_preprocessed = []
        n.sf_preprocessed = n.sf # restore original sampling frequency

        if filter_type == "Bandpass"
            rt = Bandpass( freqs_n[1], freqs_n[2] );
            dm = getfield( Main, Symbol(filter_design[1]) )(filter_order[1]);
            f = digitalfilter(rt, dm);
            ### loop over channels and filter trace
            for trc in n.trace
                #trc_padded = flatten([[.0 for ii in 1:pads], trc, [.0 for ii in 1:pads]])
                #push!(n.trace_preprocessed, filtfilt(f, trc_padded)[pads+1:end-pads]);
                push!(n.trace_preprocessed, filtfilt(f, trc));
            end
        elseif filter_type == "Lowpass"
            rt = Lowpass( freqs_n[1] );
            dm = getfield( Main,Symbol(filter_design[1]) )(filter_order[1]);
            f = digitalfilter(rt, dm);
            for trc in n.trace
                # trc_padded = flatten([[.0 for ii in 1:pads], trc, [.0 for ii in 1:pads]])
                # push!(n.trace_preprocessed, filtfilt(f, trc_padded)[pads+1:end-pads]);
                push!(n.trace_preprocessed, filtfilt(f, trc));
            end
        elseif filter_type == "Highpass"
            rt = Highpass( freqs_n[1] );
            dm = getfield( Main,Symbol(filter_design[1]) )(filter_order[1]);
            f = digitalfilter(rt, dm);
            for trc in n.trace
                # trc_padded = flatten([[.0 for ii in 1:pads], trc, [.0 for ii in 1:pads]])
                # push!(n.trace_preprocessed, filt(f, trc_padded)[pads+1:end-pads]);
                push!(n.trace_preprocessed, filtfilt(f, trc));
            end
        elseif filter_type == "Notch"
            for trc in n.trace
                trc = flatten(trc)
                for ff in freqs_n.*nyq
                    nn = iirnotch(ff, bandwidth[1]; fs=n.sf)
                    trc = filtfilt(nn, trc);
                end
                push!(n.trace_preprocessed, trc);
            end
        elseif filter_type == "CommonAverage"
            chnnls = freqs
            average_trace = [median([n.trace[ch][idx] for ch in chnnls]) for idx in 1:length(n.trace[1])]
            # n.trace_preprocessed = [channel .- average_trace for channel in n.trace[from_chnnls]]
			[n.trace_preprocessed[ch] = n.trace_preprocessed[ch] .- average_trace for ch in from_chnnls]
            #n.trace_preprocessed = [ch .- mean(n.trace) for ch in n.trace]
        end
        n.preprocessing_steps = [string( filter_type, ": ",freqs )];

    ######################
    ### preprocessed trace
    ######################
    elseif what == "trace_preprocessed" && !isempty(n.preprocessing_steps)
        nyq = n.sf_preprocessed/2;
        freqs_n = freqs./nyq; # normalized to nquist frequncy

        if filter_type == "Bandpass"
            rt = Bandpass( freqs_n[1], freqs_n[2] );
            dm = getfield( Main,Symbol(filter_design[1]) )(filter_order[1]);
            f = digitalfilter(rt, dm);
            for (itrc, trc) in enumerate(n.trace_preprocessed)
                # trc_padded = flatten([[.0 for ii in 1:pads], trc, [.0 for ii in 1:pads]])
                # n.trace_preprocessed[itrc] = filt(f, trc_padded)[pads+1:end-pads];
                n.trace_preprocessed[itrc] = filtfilt(f, trc);
            end
        elseif  filter_type == "Lowpass"
            rt = Lowpass( freqs_n[1] );
            dm = getfield( Main,Symbol(filter_design[1]) )(filter_order[1]);
            f = digitalfilter(rt, dm);
            for (itrc, trc) in enumerate(n.trace_preprocessed)
                # trc_padded = flatten([[.0 for ii in 1:pads], trc, [.0 for ii in 1:pads]])
                # n.trace_preprocessed[itrc] = filt(f, trc_padded)[pads+1:end-pads];
                n.trace_preprocessed[itrc] = filtfilt(f, trc);
            end
        elseif filter_type == "Highpass"
            rt = Highpass( freqs_n[1] );
            dm = getfield( Main,Symbol(filter_design[1]) )(filter_order[1]);
            f = digitalfilter(rt, dm);
            for (itrc, trc) in enumerate(n.trace_preprocessed)
                # trc_padded = flatten([[.0 for ii in 1:pads], trc, [.0 for ii in 1:pads]])
                # n.trace_preprocessed[itrc] = filt(f, trc_padded)[pads+1:end-pads];
                n.trace_preprocessed[itrc] = filtfilt(f, trc);
            end
        elseif filter_type == "Notch"
            for (itrc, trc) in enumerate(n.trace_preprocessed)
                trc = flatten(trc)
                for ff in freqs_n.*nyq
                    nn = iirnotch(ff, bandwidth[1]; fs=n.sf_preprocessed)
                    trc = filtfilt(nn, trc);
                end
                n.trace_preprocessed[itrc] = trc;
            end
        elseif filter_type == "CommonAverage"
            chnnls = freqs
            average_trace = [median([n.trace_preprocessed[ch][idx] for ch in chnnls]) for idx in 1:length(n.trace_preprocessed[1])]
            # n.trace_preprocessed = [channel .- average_trace for channel in n.trace_preprocessed]
			[n.trace_preprocessed[ch] = n.trace_preprocessed[ch] .- average_trace for ch in from_chnnls]
			#n.trace_preprocessed = [ch .- mean(n.trace_preprocessed) for ch in n.trace_preprocessed]
        end
        push!(n.preprocessing_steps, string( filter_type, ": ", freqs ));
    else
        print(n.preprocessing_steps)
        @warn "SELF.trace_preprocessed is empty. Process SELF.trace first"
    end
end




















"""
Arguments:

+ n::Neurons.recording
+ chans::Array{Int64,1}
+ SNR::Array{Float64,1}
+ lockout::Int64=30;
+ polarity::String="negative",
+ ww::Array{Int64,1}=[5,15],
+ stp::Int64=1,
+ w_lockout::Array{Int64,1}=[10],
+ method::String="petrantonakis"

"""
function detect_spikes(
	n::Neurons.recording,
	chans::Array{Int64,1},
	SNR::Array{Float64,1},
	lockout::Int64=30;
	#
	polarity::String="negative",
	ww::Array{Int64,1}=[5,15],
	stp::Int64=1,
	w_lockout::Array{Int64,1}=[10],
	method::String="petrantonakis")
#
#   spikes are stored as:s
#   n.spike_idx: channel X w X Dw/indices of peaks (Dw only if method is "petrantonakis")
#
#   w: window length if method is "petrantonakis"
#   (for consistency if method is "threshold" the same structure is stored )


### WAVEFORM/THRESHOLD BASED SPIKE DETECITON
    if method == "petrantonakis" ### ref: https://www.frontiersin.org/articles/10.3389/fnins.2015.00452/full
        # Dw = [ [[] for ii in 1:n.n_channels] for jj in eachindex(ww) ] # ww X channel
        # Spks = [ [CartesianIndex{1}[] for ii in 1:n.n_channels] for jj in eachindex(ww) ] # ww X channel
        Dw_spk = [ [[[],[]] for ii in eachindex(ww)] for jj in 1:n.n_channels ] ### ch X w X Dw,spk_idx
		n.detection_threshold = []

        # for (ii,ch) in enumerate([n.trace_preprocessed[2]]) ### loop over channels
		for ii in chans ### loop over channels

			ch = n.trace_preprocessed[ii]

			for (jj,w) in enumerate(ww) ### loop over w

                ### compute the Dw signal (euklidean distance signal)
                f = ch[collect(0:w-1).+collect(1:stp:length(ch)-w+1)'] ### cut signal in overlapping chunks
				Dw_spk[ii][jj][1] = flatten(sqrt.(sum(diff(f, dims=2).^2, dims=1))); ### compute euclidean distance between consecutive chunks

                ### normalization of Dw
				sig = median( Dw_spk[ii][jj][1] ./ 0.6745 )
				thr = SNR[jj]*sig
				append!(n.detection_threshold, thr)
                # Dw_spk[ii][jj][1] = Dw_spk[ii][jj][1] ./ th ### store the standardized Dw signal
                ### find all local mixima in Dw
                loc_maxima = findlocalmaxima(Float64.(Dw_spk[ii][jj][1])) ### find all local maxima
                ### remove local mixima < thr
                Dw_spk[ii][jj][2] = loc_maxima[findall(Dw_spk[ii][jj][1][loc_maxima] .> thr)] ### extract all local maxima > SNR, SNR depends on w

                ### remove redundant triggered spikes and keep the maximum of each
                if length(Dw_spk[ii][jj][2]) > 1

					Dw_spk[ii][jj][2] = find_redundant( Dw_spk[ii][jj][1], Dw_spk[ii][jj][2], lockout );

				end

			end #for w: jj

            ### remove spikes that are detected only by one value of w
			loop_cntr = 0
			while length(Dw_spk[ii][1][2]) != length(Dw_spk[ii][2][2]) ### necessary if there a spikes in value of w with less spikes that are not in w with more spikes

				loop_cntr += 1

				if loop_cntr == 10 ### 2 should be enough

					break;

				end

				if !any( [(length(Dw_spk[ii][1][2])==0), (length(Dw_spk[ii][2][2])==0)] )

					if length(Dw_spk[ii][1][2]) > length(Dw_spk[ii][2][2])

						dd = pairwise(Euclidean(), cartesian2matrix(Dw_spk[ii][2][2])', cartesian2matrix(Dw_spk[ii][1][2])', dims=2)
					    deleteat!(Dw_spk[ii][1][2], findall(flatten(.!any(dd .< w_lockout, dims=1)))) ### delete all spikes in Dw_spk[ii][1][2] that do not have a partner spike in Dw_spk[ii][2][2]

					elseif length(Dw_spk[ii][1][2]) < length(Dw_spk[ii][2][2])

						dd = pairwise(Euclidean(), cartesian2matrix(Dw_spk[ii][1][2])', cartesian2matrix(Dw_spk[ii][2][2])', dims=2)
					    deleteat!(Dw_spk[ii][2][2], findall(flatten(.!any(dd .< w_lockout, dims=1))))

					end

				else ### if one has no spikes at all

					Dw_spk[ii][1][2] = [];
					Dw_spk[ii][2][2] = [];
					# Dw_spk[ii][1][1] = []
					# Dw_spk[ii][2][1] = []

				end

			end

			if length(Dw_spk[ii][1][2]) != length(Dw_spk[ii][2][2])

				@info "@Channel ", ii, ": different spike numbers for values of w"

			end

			### scatter plot of w1 X w2
	        if (length(spk_idx_w1)-length(spk_idx_w1) == 0) && length(recording.spike_idx[channel][1][2]) != 0

	            ax1 = subplot(1,3,1)
	            w1 = recording.spike_idx[channel][1][1][spk_idx_w1]
	            w2 = recording.spike_idx[channel][2][1][spk_idx_w2]
	            pols = cartesian2polar(w1, w2)
	            r = pols[:,1]
	            phi = pols[:,2]
	            ax1.scatter( r, phi, color=pcols[2])

	        end

        end #for channels: ii

    ### THRESHOLD BASED SPIKE DETECTION
    elseif method == "threshold"

		Dw_spk = [ [[[],[]] for ii in [1]] for jj in 1:n.n_channels ] ### ch X w X DW,spk_idx
		n.detection_threshold = ones(n.n_channels) .* NaN
        x = n.trace_preprocessed[chans]

        sig = [median(abs.(ch) ./ 0.6745 ) for ch in x]
		thr = SNR[1] .* sig
		n.detection_threshold[chans] = thr

        # x = .-x ./ th

        if polarity == "both"

			ind_max = findlocalmaxima.(x)
            ind_max = [[lm[1] for lm in ch] for ch in ind_max] # cartesian coordinates to indices
            ind_min = findlocalminima.(x)
            ind_min = [[lm[1] for lm in ch] for ch in ind_min]
            ind_max = [ind_max[ii][findall(x[ii][ind_max[ii]] .> thr[ii])] for ii in eachindex(chans)]    # local minima with minimal SNR
            ind_min = [ind_min[ii][findall(x[ii][ind_min[ii]] .< -thr[ii])] for ii in eachindex(chans)]    # local maxima with minimal SNR
            ind = [sort(flatten([ind_max[ii], ind_min[ii]])) for ii in 1:length(ind_max)] # channel X index_of_spikes

		elseif polarity == "positive"

			ind = findlocalmaxima.(x)
            ind = [[lm[1] for lm in ch] for ch in ind] # cartesian coordinates to indices
            ind = [ind[ii][findall(x[ii][ind[ii]] .> thr[ii])] for ii in eachindex(chans)]    # local minima with minimal SNR

		elseif polarity == "negative"

			ind = findlocalminima.(x)
            ind = [[lm[1] for lm in ch] for ch in ind] # cartesian coordinates to indices
            ind = [ind[ii][findall(x[ii][ind[ii]] .< -thr[ii])] for ii in eachindex(chans)]

		end

        ### remove redundantly triggered spikes and store the maximum respectively
        for (ii,ch) in enumerate(chans)

			chans_spike_idxs = ind[ii]
    		Dw_spk[ch][1][2] = CartesianIndex.( find_redundant( x[ii], chans_spike_idxs, lockout ) )

		end

    end # method

    ### store clean indices in struct
    # n.spike_idx = ind_clean#[ind_clean[ii][d[ii]] for ii in 1:length(ind_clean)]
    n.spike_idx = Dw_spk; # channel X w X Dw/index of peak (Dw only if method is "petrantonakis")
end




"""
Documentation ...
"""
function extract_isih(n::Neurons.recording)
### Extracts the inter-spike intervals
    dt = 1/n.sf_preprocessed
    isih = []
    for ch in  eachindex(n.spike_idx)
        push!(isih, [])
        for w in eachindex(n.spike_idx[1])
            push!(isih[ch], [])
            if length(n.spike_idx[ch][w][2]) > 1
                isih[ch][w] = cartesian2matrix(diff(n.spike_idx[ch][w][2])) .* dt
            end
        end
    end
    n.isih = isih # value_source X w X time_source X time/values
end




"""
Documentation ...
"""
function extract_spikes(n::Neurons.recording, mp, w::Int64=1)
### Extract spike spikeforms and time stamps in ms.
#   extracts the spikeforms at indices n.spike_idx (normalized around zero)
#   from the filtered signal x using a fixed window around the
#   window +-mp of the spikes.
#	ONLY FOR ONE VALUE OF w
#   structure: n.spikeforms[val_source][time_source][spike_time/value]
#

    dt = 1/n.sf_preprocessed
	window = collect(mp[1]:1:mp[2]);
	spkfrms = []

	for ii in 1:n.n_channels # loop over channels as value source

		push!(spkfrms, [])
	    # for w in 1:size(n.spike_idx[1],1)
	    #     push!(spkfrms[ii], [])

			for (jj,spk_idxs) in enumerate(n.spike_idx) # loop over channels for time source

				push!(spkfrms[ii], [[],[]]) ### t, s

				for spk in spk_idxs[w][2] # loop over individual spikes

				    # skip spikes that have negative indices
	                spike_window_idxs = spk.I .+ window # generate window for each spike
	                spike_window_times = spike_window_idxs .* dt .* 1000 # spike times in ms

					if spike_window_idxs[1] > 0 && spike_window_idxs[end] <= length(n.trace_preprocessed[ii])# skip spikes that are not complete in window

						push!( spkfrms[ii][jj][1], spike_window_times ) # store spike times
	                    push!( spkfrms[ii][jj][2], n.trace_preprocessed[ii][spike_window_idxs] .- mean( n.trace_preprocessed[ii][spike_window_idxs]) ) #store spike values

					end
	            end
	        end
	    # end
	end
	n.spikeforms = spkfrms # value_source X w X time_source X time/values

end





"""
Documentation ...
"""
function extract_features(n::Neurons.recording)
    n.features = []
    for ch in 1:n.n_channels
        push!(n.features, [])
        if !isempty(n.spikeforms[ch][ch][2])
            ### make matrix for PCA
            m = zeros(length(n.spikeforms[ch][ch][2]), length(n.spikeforms[ch][ch][2][1])); # #spikes X #samples per spike
            [m[ii,:] = n.spikeforms[ch][ch][2][ii] for ii = 1:length(n.spikeforms[ch][ch][2])]; # spike X sample
            ### fit PCA
            M = fit(PCA, m)
            n.features[ch] = projection(M) # projection(M): spike X PC
        end
    end
end



"""
Documentation ...
"""
function cluster_spikes(n::Neurons.recording, algorithm::AbstractArray, init_method::AbstractArray, feats::AbstractArray, n_clusters::AbstractArray, nIter_EM::AbstractArray, nIter_kmeans_init::AbstractArray, cov_kind::AbstractArray)

############
# cluster given features using _algorithm_
# BIC optimization for cluster number
#
########
# algorithm: kmeans
#
########
# algorithm: gmm
# init_method: :split or :kmeans

    n.spike_class = []
    ###
    if algorithm[1] == "kmeans"
        for ch in 1:n.n_channels
            #push!(n.spike_class, [])
            if size(n.features[ch], 2) > feats[end] # if there are enough features
                result = kmeans(n.features[ch][:,feats]', n_clusters[1]);
                clusters = []
                for cl = 1:n_clusters[1]
                    push!(clusters, findall(result.assignments .== cl))
                end
                push!(n.spike_class, clusters)
            else ### push empty array for that channel
                push!(n.spike_class, [])
            end
        end
    ###
    elseif algorithm[1] == "gmm"
        n.BIC = []
        ### for each channel fit gmm with n components specified in
        ### n_clusters, compute BIC fear each value, find minimum BIC and use
        ### for clustering
        for ch in 1:n.n_channels
            if size(n.features[ch],2) > length(feats) ### if enough features
                N = size(n.features[ch][:,feats], 1)
                BICS = Array{Float64,2}(undef, 0, 2)
                for cmpnts in n_clusters
                    if size(n.features[ch], 1) > cmpnts
                        M = GMM(cmpnts, n.features[ch][:,feats];
                            method=init_method[1], kind=cov_kind[1], nInit=nIter_kmeans_init[1], nIter=nIter_EM[1], nFinal=nIter_EM[1])
                        LL = sum(llpg(M, n.features[ch][:,feats]))
                        par = cmpnts + cmpnts*length(feats) + cmpnts* length(feats)*(length(feats)+1)/2 # number of parameters
                        bic = -2*LL + log.(N)*par
                        BICS = vcat(BICS, [cmpnts bic])
                    end
                end
                push!(n.BIC, BICS)
                ### fit model with optimal number of components and cluster spikes
                opt_cmpnts = Int(BICS[findmin(BICS[:,2])[2]])
                push!(n.spike_class, [])
                M = GMM(opt_cmpnts, n.features[ch][:,feats];
                    method=init_method[1], kind=cov_kind[1], nInit=nIter_kmeans_init[1], nIter=nIter_EM[1], nFinal=nIter_EM[1])
                posteriors = gmmposterior( M, n.features[ch][:,feats] )
                classes = [findmax(posteriors[1], dims=2)[2][spk][2] for spk in 1:length(findmax(posteriors[1], dims=2)[2])] ### extract cluster index for each spike from CartesianCoordinates
                n.spike_class[ch] = [[] for ii in 1:opt_cmpnts]
                for (spk_idx,cl) in enumerate(classes)
                    push!(n.spike_class[ch][cl], spk_idx)
                end
            else
                push!(n.spike_class, [])
                push!(n.BIC, [])
            end
        end
        # for ch in 1:n.n_channels
        #     print(ch)
        #     push!(n.spike_class, [])
        #     if size(n.features[ch],2) > length(feats) ### if enough features
        #         M = GMM(n_clusters[1], n.features[ch][:,feats];
        #             method=:kmeans, kind=cov_kind[1], nInit=nIter_kmeans_init[1], nIter=nIter_EM[1], nFinal=nIter_EM[1])
        #         posteriors = gmmposterior( M, n.features[ch][:,feats] )
        #         classes = [findmax(posteriors[1], dims=2)[2][spk][2] for spk in 1:length(findmax(posteriors[1], dims=2)[2])] ### extract cluster index for each spike from CartesianCoordinates
        #         n.spike_class[ch] = [[] for ii in 1:n_clusters[1]]
        #         for (spk_idx,cl) in enumerate(classes)
        #             push!(n.spike_class[ch][cl], spk_idx)
        #         end
        #     else
        #         n.spike_class[ch] = [[] for ii in 1:n_clusters[1]]
        #     end
        # end
    end
end






"""
Up- or downsampling
Acts on either n.trace or n.trace_preprocessed
"""
function resample(n::Neurons.recording, frequency::Real)
    #idx = 1:factor:size(n.trace)[1]
    #n.trace_preprocessed = n.trace[idx];
    #n.sf_decimated = n.sf/factor;
    if isempty(n.preprocessing_steps)
        n.trace_preprocessed = DSP.resample( n.trace,frequency//convert(Int64, n.sf) );
    else
        n.trace_preprocessed = DSP.resample( n.trace_preprocessed,frequency//convert(Int64, n.sf_preprocessed) );
    end
    n.sf_preprocessed = frequency;
    push!(n.preprocessing_steps, string( "Decimated to: ",frequency ))
end

"""
Up- or downsampling
Acts on either n.trace or n.trace_preprocessed
"""
function get_PSD(n::Neurons.recording, which, psd_type)
    if which == "trace"
        return [getfield( Main, Symbol(psd_type) )( flatten(ch), fs=n.sf ) for ch in n.trace]
    elseif which == "trace_preprocessed"
        return [getfield( Main, Symbol(psd_type) )( flatten(ch), fs=n.sf_preprocessed ) for ch in n.trace_preprocessed]
    end
end




function find_redundant(ref::AbstractArray, ind::AbstractArray, lockout::Int)
    ### ref: 1D array containing amplitude like values
    ### ind: 1D array containing indices of identified events, e.g. spikes
    ### lockout: Integer defining minimum distance of evvents in samples that are treated as separate events

    ### make sure it's cartesian!
    ind = CartesianIndex.(ind)
    ### matrix with all pairs of spike indices
    # mm = Iterators.product(flatten(cartesian2matrix(ind)), flatten(cartesian2matrix(ind)))
	### differences between all pairs of spike indices
	# dd = [ float(abs(diff([x for x in m])[1][1])) for m in mm ]
	dd = pairwise(Euclidean(), flatten(cartesian2matrix(ind))', flatten(cartesian2matrix(ind))', dims=2)
	dd[diagind(dd)] .= NaN # set diagonal to NaN
	sup_diag = dd[diagind(dd, 1)]
	gg = findall(x -> x > lockout, sup_diag )

	if all(sup_diag .> lockout) ### no double triggered spikes

		groups = [[spk_idx] for spk_idx in ind]

	elseif all(sup_diag .< lockout) ### only one double triggered spike

		groups = [ind]

	else ### single and double triggered spikes

		groups = []
	    current_group = []

		for (ij,spk_id) in enumerate(ind)

			push!(current_group, spk_id)

			if in(ij, gg) || ij == length(ind) ### ij == length(ind) required for last element

				push!(groups, current_group)
	            current_group = []

			end
		end
	end
	### find maxima of groups and store indices
    ind_clean = []
    for gr in groups

		max_id = findmax( abs.(ref[CartesianIndex.( gr )]) )
		# max_id = findmax(ref[gr])
    	push!(ind_clean, gr[max_id[2]])

	end

    #################
    ### DELETE ME ###a
    #################
    # ind_clean = []
    # ind_ = deepcopy(ind)
    # if length(ind_) > 1
    #     groups = []
    #     current_group = []
    #     push!(current_group, ind_[1])
    #     deleteat!(ind_, 1)
    #
    #     ### extract groups of detected event that are too close to each other to represent seperate spikes
    #     while !isempty(ind_)
    #         ### while next data point is to close to preceeding one, add it to current group
    #         while ind_[1][1]-current_group[end][1] < lockout ### double index because CartesianIndex
    #             push!(current_group, ind_[1])
    #             deleteat!(ind_, 1)
    #             if isempty(ind_) ### if already at the last spike index
    #                 break;
    #             end
    #         end
    #
    #         ### add current_group to group
    #         push!(groups, current_group)
    #         current_group = []
    #         if !isempty(ind_) ### deals with if last index was already deleted in while loop
    #             push!(current_group, ind_[1])
    #             deleteat!(ind_, 1)
    #         end
    #     end
    #
    #     ### for each group extract the maximum as representation
    #     ind_c = []
    #     for gr in groups
    #         if length(gr) > 1 ### if the group is a "group"
    #             gg = [gr[ii] for ii in eachindex(gr)]
    #             push!( ind_c, gg[findmax(ref[gg])[2]] )
    #         else ### if the group has only one entry
    #             push!(ind_c, gr[1])
    #         end
    #     end
    #     push!(ind_clean, ind_c)
    # else ### if no spikes in channel, add empty list to kepp size consistent
    #     push!(ind_clean, [])
    # end

    return ind_clean
end



################
### PLOTTING ###
################

function plot_detection(
        recording::Neurons.recording;
        channel::Int64,
        method::String="petrantonakis",
        polarity::String="negative")

	colors = myColors() # from MyHelpers.jl
	pcols = map(col -> (red(col), green(col), blue(col)), colors)

    if method == "petrantonakis"

        fig = figure(figsize=(13,5))
        rc("xtick",labelsize=8)
        rc("ytick",labelsize=8)

		### time course of Dw signal
        Dw_w1 = recording.spike_idx[channel][1][1]
        Dw_w2 = recording.spike_idx[channel][2][1]
		### indices of spikes
		spk_idx_w1 = recording.spike_idx[channel][1][2]
        spk_idx_w2 = recording.spike_idx[channel][2][2]
		### spike indices from cartesian to matrix
		if !isempty(spk_idx_w1) && !isempty(spk_idx_w2)
			spk_idx_w1 = cartesian2matrix(spk_idx_w1)
			spk_idx_w2 = cartesian2matrix(spk_idx_w2)
		end
        w1_threshold = recording.detection_threshold[1]
        w2_threshold = recording.detection_threshold[2]

        ### scatter plot of w1 X w2
        if (length(spk_idx_w1)-length(spk_idx_w1) == 0) && length(recording.spike_idx[channel][1][2]) != 0

            ax1 = subplot(1,3,1)
            w1 = recording.spike_idx[channel][1][1][spk_idx_w1]
            w2 = recording.spike_idx[channel][2][1][spk_idx_w2]
            pols = cartesian2polar(w1, w2)
            r = pols[:,1]
            phi = pols[:,2]
            ax1.scatter( r, phi, color=pcols[2])

        end

        ax2 = subplot(1,3,2)
        ax2.plot( Dw_w1, color=pcols[1], zorder=1 )
        ax2.plot( 1:length(Dw_w1), ones(length(Dw_w1))*w1_threshold, color=:red, linestyle="--" )
        if length(recording.spike_idx[channel][1][2]) > 0

            spk_idx_w1 = cartesian2matrix(recording.spike_idx[channel][1][2])
            ax2.scatter(spk_idx_w1, Dw_w1[spk_idx_w1], color=pcols[2], zorder=2)

        end

        ax3 = subplot(1,3,3)
        ax3.plot(Dw_w2, color=pcols[1], zorder=1)
        ax3.plot( 1:length(Dw_w2), ones(length(Dw_w2))*w2_threshold, color=:red, linestyle="--" )
		if length(recording.spike_idx[channel][2][2]) > 0

            spk_idx_w2 = cartesian2matrix(recording.spike_idx[channel][2][2])
            ax3.scatter(spk_idx_w2, Dw_w2[spk_idx_w2], color=pcols[2], zorder=2)

        end

    elseif method == "threshold"

        fig = figure(figsize=(13,5))
        rc("xtick",labelsize=8)
        rc("ytick",labelsize=8)

        x = recording.trace_preprocessed[channel]
        spk_idx = cartesian2matrix(recording.spike_idx[channel][1][2])

		if polarity == "negative"

			thr = -recording.detection_threshold[channel]

		elseif polarity == "positive"

			thr = recording.detection_threshold[channel]

		elseif polarity == "both"

			thr = [-recording.detection_threshold[channel], recording.detection_threshold[channel]]
        end

        plot( x )
        scatter( spk_idx, x[spk_idx], color=pcols[2], zorder=2 )

        if length(thr) != 2

			plot( 1:length(x), ones(length(x))*thr, color=:red, linestyle="--" )

		else

			plot( 1:length(x), ones(length(x))*thr[1], color=:red, linestyle="--" )
            plot( 1:length(x), ones(length(x))*thr[2], color=:red, linestyle="--" )
        end
    end
end


function plot_waveforms(
	recording::Neurons.recording,
	t::Array{Float64,1},
	ylims::Array{Int64,2},
	w::Int64=1;
	#
	t_source )
	##########################
	##########################
	##########################

	colors = myColors() # from MyHelpers.jl
	pcols = map(col -> (red(col), green(col), blue(col)), colors)
	half_window = abs(minimum(t))/1000

	fig = figure(figsize=(15,7), dpi=300)
	rc("xtick",labelsize=4)
	rc("ytick",labelsize=4)

	# ylims = [-1000 1000]

	if t_source == "val_source"
		corresponding = true
	else
		corresponding = false
	end

	for val_source = 1:recording.n_channels
	    #### t_cource determines whose channels spike times are used to cut out the the waveforms: if t_cource = val_source then channel and spiketimes are matched, i.e. 1<->1, 2<->2 etc
		if corresponding
			t_source = val_source
		end
	#     t_source = val_source

	    ### sort spikes according to amplitude
	    sorted_idx = sortperm([maximum(spk) for spk in recording.spikeforms[val_source][t_source][2]] .- [minimum(spk) for spk in recording.spikeforms[val_source][t_source][2]], rev=true )
	    ### how many spikes to plot
	    # n = length(sorted_idx)
	    n = 10

	    if !isempty(sorted_idx) ### if there are spikes in sorted_idx, i.e. if channel has spikes

	        spks_to_plot = []

	        ### loop over clusters of current channel
	        if !isempty(recording.spike_class)
	            for cl in recording.spike_class[t_source] ### plot the clusters based on the t_source channel
	                push!(spks_to_plot, intersect(cl, sorted_idx))
	            end
	        else
	            push!(spks_to_plot, sorted_idx) ### i.e. one "cluster" if not yet clustered
	        end


	        ### check if there are enough spikes if number of spikes to plot is specified
	        if length(sorted_idx) < n
	            n = length(sorted_idx)
	        end

	        ### plot spikes belonging to clusters
			offset = 0
	        for (ii,spk_cl) in enumerate(spks_to_plot)
	            if !isempty(spk_cl) ### cluster is not empty
	                subplot(2,5,val_source)
				   ### plot individual traces
				   plot(t, reduce(hcat, recording.spikeforms[val_source][t_source][2][spk_cl]),
					  color=pcols[ii], alpha=0.3, linewidth=0.1, zorder=1) ### plot individual spikes
				   ### plot mean waveform +- std
				   clusters_mean = mean(reduce(hcat, recording.spikeforms[val_source][t_source][2][spk_cl]), dims=2) .+ offset
				   clusters_std = std(reduce(hcat, recording.spikeforms[val_source][t_source][2][spk_cl]), dims=2)
				   plot(t, clusters_mean, color=pcols[ii], alpha=0.5, linewidth=0.5, zorder=2) ### plot mean spike waveform of cluster
				   fill_between(t, flatten(clusters_mean+clusters_std), flatten(clusters_mean-clusters_std), color=pcols[ii], alpha=0.2, linewidth= 0.1)
				   ###
				   title(string("Channel ", val_source, " (", n, ")"), fontsize=5)
				   ylim(ylims[1], ylims[2])
	# 			   legend()
	            end
				offset = offset +0
	        end

	        ### add x- and ylabel
	        if val_source in collect(6:10)
	            subplot(2,5,val_source)
	            xlabel("ms", fontsize=5)
	        elseif val_source in [1,6]
	            subplot(2,5,val_source)
	#             ylabel("amplitude")
	        end
			if val_source in collect(1:5)
			   xticks(-half_window*1000:1:half_window*1000, []) ### half_window comes from waveform extraction step
			end
			if val_source in [2:5 7:10]
			   yticks(ylims[1]:100:ylims[2], [])
			end

	    else ### plot empty axes of channel has no spikes
	        subplot(2,5,val_source)
	        plot(t, t,
	            linewidth=0)
	        title(string("Channel ", val_source, " (", n, ")"))
	        ### add x- and ylabel
	        ### add x- and ylabel
	        if val_source in collect(6:10)
	            subplot(2,5,val_source)
	            xlabel("ms", fontsize=5)
	        elseif val_source in [1,6]
	            subplot(2,5,val_source)
	#             ylabel("amplitude")
	        end
			if val_source in collect(1:5)
			   xticks(-4:2:4, [])
			end
			if val_source in [2:5 7:10]
			   yticks(ylims[1]:100:ylims[2], [])
			end

	    end

	end

end






###
end
