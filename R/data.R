#' International Personality Item Pool (IPIP) NEO personality inventory (Johnson, 2014).
#'
#' This inventory is a public-domain version of the widely used NEO personality inventory (Costa & McCrae, 1985), 
#' which is designed to measure the big five personality factors, including Neuroticism (N), Agreeableness (A), 
#' Extraversion (E), Openness to experience (O), and Conscientiousness (C). According to (Johnson, 2014), each
#' personality factor can be further split into six personality facets, resulting in 30 facets.
#'
#' @format A list with the following elements:
#' \describe{
#'   \item{response}{0-4 response data with 7325 rows and 300 items, categorical scale}
#'   \item{Q}{Q matrix specifying the non-zero elements in the loading matrix}
#'   \item{J}{# of items}
#'   \item{K}{# of latent traits}
#'   \item{M}{# of response categories}
#'   \item{N}{sample size}
#' }
#' @source \url{https://osf.io/wxvth/}
"ipip300_complete"