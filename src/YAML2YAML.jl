function parse_highyml_to_lowyml(
    highyml::Union{AbstractDict,AbstractString}
    ; basedir::String = pwd(),
      include_subdirectory ::Union{Nothing,Bool} = nothing,
      roi ::Union{Nothing,DetectorAxes} = nothing,
      title ::Union{Nothing,String} = nothing,
      fixed_file_list ::Union{Nothing,Vector{String}} = nothing,
      generated ::DateTime= Dates.now()
) ::OrderedDict{String,Any}

    # use can give a filepath to yaml file, or an already parsed yaml (as a Dict)
    if highyml isa AbstractString
        highyml = YAML.load_file(highyml)
    end

    haskey(highyml, "exptime")     || throw(ArgumentError("yaml dict miss key \"exptime\""))
    haskey(highyml, "categories")  || throw(ArgumentError("yaml dict miss key \"categories\""))
    for (catname,catdef) in highyml["categories"]
        if !haskey(catdef, "sources")
            throw(ArgumentError("Category \"$catname\" dict miss key \"sources\""))
        end
    end
    
    # keyword julia parameter has priority over yaml parameter
    if isnothing(title)
        if haskey(highyml, "title")
            title = highyml["title"]
        else
            title = "YAML file produced by ScientificDetectors"
        end
    end
    
    # keyword julia parameter has priority over yaml parameter
    if isnothing(roi)
        if haskey(highyml, "roi")
            roi = highyml["roi"]
        else
            throw(ArgumentError("yaml dict miss key \"roi\""))
        end
    else
        roi = [ OrderedDict("offset" => axis.off,
                            "length" => axis.len,
                            "step"   => axis.stp,
                            "bin"    => axis.bin)  for axis in roi ]
    end
    
    exptimekwd = highyml["exptime"]

    suffixes = haskey(highyml, "suffixes") ? highyml["suffixes"] : [".fits", ".fits.gz", ".fits.Z"]

    # keyword julia parameter has priority over yaml parameter
    if isnothing(include_subdirectory)
        if haskey(highyml, "include subdirectory")
            include_subdirectory = highyml["include subdirectory"]
        else
            include_subdirectory = false
        end
    end
    
    lowyml = OrderedDict{String,Any}(
        "title"      => title,
        "generated"  => generated,
        "roi"        => roi,
        "categories" => OrderedDict{String,Any}()
    )
    
    # get the list of paths of every FITS file inside `basedir`
    # unless the caller gave a fixed file list, in that case we just use that list
    fitspaths = if isnothing(fixed_file_list)
                    get_fits_paths(basedir; include_subdirectory, suffixes)
                else
                    fixed_file_list
                end

    # for each file, we put it in a category if it respects the category's conditions
    # these categories conditions are described in `highyml["categories"]`
    for fitspath in fitspaths
        FitsFile(fitspath) do fits
            for (catname,catdef) in highyml["categories"]

                # check that this file respect every keyword condition listed in higyml category
                valid = true
                for (keyword,values) in catdef
                    if keyword == "sources"
                        continue
                    end
                    if haskey(fits[1], keyword)
                        if (values isa Vector) && any(==(fits[1][keyword].value()), values)
                            # good, the file respects this condition
                        elseif !(values isa Vector) && (values == fits[1][keyword].value())
                            # good, the file respects this condition
                        else
                            valid = false
                            break
                        end
                    else
                        valid = false
                        break
                    end
                end
                
                if valid
                    # create category if non existing already
                    if !haskey(lowyml["categories"], catname)
                        lowyml["categories"][catname] = OrderedDict{String,Any}(
                            "sources" => catdef["sources"],
                            "files"   => OrderedDict{Float64,Any}()
                        )
                    end
                    
                    file_exptime =
                        if haskey(fits[1], exptimekwd)
                            fits[1][exptimekwd].float
                        else
                            throw("Fits file \"$fitspath\" miss exptime keyword \"exptimekwd\"")
                        end
                    
                    # create entry for that exptime if it doest not exist yet
                    if !haskey(lowyml["categories"][catname]["files"], file_exptime)
                        lowyml["categories"][catname]["files"][file_exptime] = Vector{String}()
                    end

                    # add filepath to this category to this exptime
                    push!(lowyml["categories"][catname]["files"][file_exptime], fitspath)
                end
            end
        end
    end
    
    sort!(lowyml["categories"])
    for (catname,catdef) in lowyml["categories"]
        sort!(catdef["files"])
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