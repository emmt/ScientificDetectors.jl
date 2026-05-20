function categorize_calib_files(
    high_yml::Union{AbstractDict,AbstractString},
    calib_files_dir::String = pwd()
    ; roi ::Union{Nothing,DetectorAxes} = nothing,
      datahdu ::Union{Nothing,Integer,String} = nothing,
      title ::Union{Nothing,String} = nothing,
      include_subdirectories ::Union{Nothing,Bool} = nothing,
      suffixes ::Union{Nothing,Vector{String}} = nothing,
      fixed_file_list ::Union{Nothing,Vector{String}} = nothing
) ::OrderedDict{String,Any}

    # use can give a filepath to yaml file, or an already parsed yaml (as a Dict)
    if high_yml isa AbstractString
        high_yml = YAML.load_file(high_yml; dicttype=OrderedDict{String,Any})
    end
    
    low_yml = OrderedDict{String,Any}()
    
    low_yml["title"] = something(title,
                                get(high_yml, "title", "YAML file produced by ScientificDetectors"))

    low_yml["generated"] = Dates.now()

    low_yml["roi"] = if isnothing(roi)
                        deepcopy(high_yml["roi"])
                    else
                        [ OrderedDict("offset" => axis.off,
                                      "length" => axis.len,
                                      "step"   => axis.stp,
                                      "bin"    => axis.bin)
                          for axis in roi ]
                    end
    
    low_yml["datahdu"] = something(datahdu, get(high_yml, "datahdu", 1))
    
    low_yml["categories"] = OrderedDict{String,Any}()
    for (catname,cat) in high_yml["categories"]
        low_yml["categories"][catname] = OrderedDict{String,Any}()
        low_yml["categories"][catname]["sources"] = deepcopy(cat["sources"])
        low_yml["categories"][catname]["files"] = OrderedDict{Float64,Vector{String}}()
        if haskey(cat, "datahdu")
            low_yml["categories"][catname]["datahdu"] = cat["datahdu"]
        end
    end

    exptimekwd = high_yml["exptime keyword"]

    suffixes = something(suffixes, get(high_yml, "suffixes", [".fits", ".fits.gz", ".fits.Z"]))

    include_subdirectories = something(include_subdirectories,
                                     get(high_yml, "include subdirectories",
                                         false))

    # get the list of paths of every FITS file inside `calib_files_dir`
    # unless the caller gave a fixed file list, in that case we just use that list
    fitspaths = if isnothing(fixed_file_list)
                    get_fits_paths(calib_files_dir; include_subdirectories, suffixes)
                else
                    fixed_file_list
                end

    # for each file, we put it in a category if it respects the category's conditions
    # these categories conditions are described in `high_yml["categories"]`
    msg = "associating FITS files to calibration categories"
    @showprogress msg for fitspath in fitspaths
        FitsFile(fitspath) do fits
            H = FitsHeader(fits[1])
            for (catname,cat) in high_yml["categories"]

                # check that this file respect every keyword condition listed in higyml category
                valid = true
                for (kwd, accepted_vals) in cat
                    # skip non keyword entries
                    kwd in ("sources", "datahdu") && continue
                    # get header value for that keyword
                    haskey(H,kwd) || (valid = false; break)
                    header_val = H[kwd].value()
                    # header value must be one of the accepted values
                    if accepted_vals isa AbstractVector
                        (header_val in accepted_vals) || (valid = false; break)
                    else
                        (header_val == accepted_vals) || (valid = false; break)
                    end
                end
                
                if valid
                    if haskey(H,exptimekwd)
                        Δt = H[exptimekwd].float
                        # create entry for that exptime if it doest not exist yet
                        if !haskey(low_yml["categories"][catname]["files"], Δt)
                            low_yml["categories"][catname]["files"][Δt] = Vector{String}()
                        end
                        # add filepath to this category to this exptime
                        push!(low_yml["categories"][catname]["files"][Δt], fitspath)
                    else
                        @error "Fits file \"$fitspath\" miss exptime keyword \"$exptimekwd\"."
                    end
                end
            end
        end
    end

    # sort files by increasing exptime
    for (catname,cat) in low_yml["categories"]
        sort!(cat["files"])
    end

    low_yml
end

"""
    get_fits_paths(dirs="./"; kwds...) -> Vector{String}

Returns the path of every FITS file contained in the given dir(s).

# Arguments
- ̀`dirs ::String... = "./"`: paths to the dir(s) containing the fits files.
## Keyword arguments
- `include_subdirectories ::Bool = true`: when `true`, search in sub dirs too.
- `suffixes ::Vector{String} = [".fits", ".fits.gz", ".fits.Z"]`: to determine
  if a file is a FITS file, only the extension of the filename is used. This parameter contains the
  list of recognized extensions. Empty list means any filename is accepted.

# Examples

```jldoctest
julia> get_fits_paths("../data/science")
2-element Vector{String}:
 "../data/science/2580479/SPHER.2020-01-12T07-16-41.460.fits.Z"
 "../data/science/2580479/SPHER.2020-01-12T07-18-16.072.fits.Z"
```
"""
function get_fits_paths(
    dirs ::String... = "./",
    ; include_subdirectories ::Bool = true,
      suffixes ::Vector{String} = [".fits", ".fits.gz", ".fits.Z"],
) ::Vector{String}
    onerror(e) = @error e
    filepaths = String[]
    for start_dir in dirs
        for (dir, _, files) in walkdir(normpath(start_dir); topdown=true, onerror)
            for file in files
                if isempty(suffixes) || any(ext -> endswith(file, ext), suffixes)
                    push!(filepaths, normpath(dir, file))
                end
            end
            include_subdirectories || break # first dir is the top one in the tree
        end
    end
    return filepaths
end