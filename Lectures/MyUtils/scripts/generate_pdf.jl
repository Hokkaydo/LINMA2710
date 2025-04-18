import PlutoPDF

function generate_pdfs(dir)
    for filename in readdir(dir)
        if endswith(filename, ".jl")
            PlutoPDF.pluto_to_pdf(joinpath(dir, filename))
        end
    end
end

generate_pdfs(dirname(dirname(joinpath(@__DIR__))))
