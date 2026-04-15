function parse_highyml_to_lowyml(
    highyml::Union{AbstractDict,AbstractString},
    calib_files_dir::String = pwd()
    ; include_subdirectory ::Union{Nothing,Bool} = nothing,
      roi ::Union{Nothing,DetectorAxes} = nothing,
      datahdu ::Union{Nothing,Integer,String} = nothing,
      title ::Union{Nothing,String} = nothing,
      fixed_file_list ::Union{Nothing,Vector{String}} = nothing
) ::OrderedDict{String,Any}

    # use can give a filepath to yaml file, or an already parsed yaml (as a Dict)
    if highyml isa AbstractString
        highyml = YAML.load_file(highyml)
    end
    
    lowyml = OrderedDict{String,Any}()
    
    lowyml["title"] = something(title,
                                get(highyml, "title", "YAML file produced by ScientificDetectors"))

    lowyml["generated"] = Dates.now()

    lowyml["roi"] = if isnothing(roi)
                        deepcopy(highyml["roi"])
                    else
                        [ OrderedDict("offset" => axis.off,
                                      "length" => axis.len,
                                      "step"   => axis.stp,
                                      "bin"    => axis.bin)
                          for axis in roi ]
                    end
    
    lowyml["datahdu"] = something(datahdu, get(highyml, "datahdu", 1))
    
    lowyml["categories"] = OrderedDict{String,Any}()
    for (catname,cat) in highyml["categories"]
        lowyml["categories"][catname] = OrderedDict{String,Any}()
        lowyml["categories"][catname]["sources"] = deepcopy(cat["sources"])
        lowyml["categories"][catname]["files"] = OrderedDict{Float64,Any}()
        if haskey(cat, "datahdu")
            lowyml["categories"][catname]["datahdu"] = cat["datahdu"]
        end
    end

    exptimekwd = highyml["exptime keyword"]

    suffixes = get(highyml, "suffixes", [".fits", ".fits.gz", ".fits.Z"])

    include_subdirectory = something(include_subdirectory,
                                     get(highyml, "include subdirectory",
                                         false))

    # get the list of paths of every FITS file inside `calib_files_dir`
    # unless the caller gave a fixed file list, in that case we just use that list
    fitspaths = if isnothing(fixed_file_list)
                    get_fits_paths(calib_files_dir; include_subdirectory, suffixes)
                else
                    fixed_file_list
                end

    # for each file, we put it in a category if it respects the category's conditions
    # these categories conditions are described in `highyml["categories"]`
    for fitspath in fitspaths
        FitsFile(fitspath) do fits
            H = FitsHeader(fits[1])
            for (catname,cat) in highyml["categories"]

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
                        if !haskey(lowyml["categories"][catname]["files"], Δt)
                            lowyml["categories"][catname]["files"][Δt] = Vector{String}()
                        end
                        # add filepath to this category to this exptime
                        push!(lowyml["categories"][catname]["files"][Δt], fitspath)
                    else
                        @error "Fits file \"$fitspath\" miss exptime keyword \"$exptimekwd\"."
                    end
                end
            end
        end
    end

    # sort files by increasing exptime
    for (catname,cat) in lowyml["categories"]
        sort!(cat["files"])
    end

    lowyml
end

"""
    get_fits_paths(dirpaths="./"; kwds...) -> Vector{String}

Returns the path of every FITS file contained in the given dir(s).

# Arguments
- ̀`dirpaths ::String... = "./"`: paths to the dir(s) containing the fits files.
## Keyword arguments
- `include_subdirectory ::Bool = true`: when `true`, search in sub dirs too.
- `follow_symlinks ::Bool = true`: see `walkdir` help.
- `suffixes ::Vector{String} = (".fits", ".fits.gz", ".fits.Z")`: to determine
  if a file is a FITS file, only the extension of the filename is used. This parameter contains the
  list of recognized extensions.

# Examples

```jldoctest
julia> get_fits_paths("../data/science")
2-element Vector{String}:
 "../data/science/2580479/SPHER.2020-01-12T07-16-41.460.fits.Z"
 "../data/science/2580479/SPHER.2020-01-12T07-18-16.072.fits.Z"
```
"""
function get_fits_paths(
    dirpaths ::String... = "./",
    ; include_subdirectory      ::Bool = true,
      follow_symlinks ::Bool = true,
      suffixes ::Vector{String} = (".fits", ".fits.gz", ".fits.Z"),
) ::Vector{String}
    onerror(e) = @error e
    filepaths = String[]
    for dirpath in dirpaths
        dirpath = normpath(dirpath)
        for (currentdir, dirs, files) in walkdir(dirpath; topdown=true, follow_symlinks, onerror)
            for file in files
                if any(ext -> endswith(file, ext), suffixes)
                    push!(filepaths, normpath(currentdir, file))
                end
            end
            include_subdirectory || break # first dir is the top one in the tree
        end
    end
    return filepaths
end