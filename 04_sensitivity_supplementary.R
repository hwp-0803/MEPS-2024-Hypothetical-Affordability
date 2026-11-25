# Sensitivity and supplementary analyses (Appendix Tables A2 and A3; labelled "Table S1" and "Table S2"
# in the output) for
# "Hypothetical Medical Bill Affordability and Experienced Financial Hardship by Race and Ethnicity" (MEPS 2024, HC-256).
# These analyses were added after the main analyses to examine alternative explanations; they are exploratory.
# Usage: Rscript 04_sensitivity_supplementary.R path/to/input.dta
suppressMessages({library(haven); library(survey)})
args <- commandArgs(TRUE); stopifnot(length(args) == 1)
cat("R", R.version$major, ".", R.version$minor, " | survey", as.character(packageVersion("survey")),
    " | haven", as.character(packageVersion("haven")), "\n\n", sep = "")
v <- c("DUPERSID","DUID","FAMID42","VARSTR","VARPSU","ESAQWT24F","BORNUSA","RACETHX","EQPAYB53","PROBPY42",
       "MEDDEBT42","EQSNON53","AGE24X","SEX","POVCAT24","INSCOV24","TOTTCH24","HIDEG",
       "OBTOTV24","OPTOTV24","ERTOT24","IPDIS24","DLAYCA42","AFRDCA42")
d <- as.data.frame(zap_labels(read_dta(args[1], col_select = all_of(v))))

d$common <- d$ESAQWT24F > 0 & d$BORNUSA %in% 1:2 & d$RACETHX %in% 1:4 & d$EQPAYB53 %in% 1:7 &
  d$PROBPY42 %in% 1:2 & d$MEDDEBT42 %in% 0:7 & d$EQSNON53 %in% 1:2
d$race <- factor(d$RACETHX, levels = c(2, 1, 3, 4), labels = c("NHW", "Hisp", "NHB", "NHA"))
d$u <- as.numeric(d$EQPAYB53 == 7)
d$probpy <- as.numeric(d$PROBPY42 == 1); d$debt <- as.numeric(d$MEDDEBT42 >= 1); d$sac <- as.numeric(d$EQSNON53 == 2)
d$age <- (d$AGE24X - 50) / 20; d$female <- as.numeric(d$SEX == 2)
d$fb <- factor(d$BORNUSA); d$pov <- relevel(factor(d$POVCAT24), ref = "5"); d$ins <- factor(d$INSCOV24)

# Total health care charges (fully established charges, excluding prescribed medicines; TOTTCH24),
# for the person and summed over all persons in the person's family (DUID + FAMID42) on the file.
d$fam <- paste(d$DUID, d$FAMID42, sep = "_")
d$famchg <- ave(d$TOTTCH24, d$fam, FUN = sum)
d$lchg_own <- log10(1 + d$TOTTCH24); d$lchg_fam <- log10(1 + d$famchg)
# Highest degree at MEPS entry (HIDEG): 1-2 no degree/GED, 3 high school diploma, 7 other degree, 4-6 bachelor's or higher
d$edu <- factor(ifelse(d$HIDEG %in% 1:2, "lt_hs", ifelse(d$HIDEG == 3, "hs", ifelse(d$HIDEG == 7, "other",
                ifelse(d$HIDEG %in% 4:6, "ba", NA)))), levels = c("ba", "lt_hs", "hs", "other"))
# Health care use in 2024: any office-based, outpatient, emergency department, or inpatient event
d$visit <- as.numeric(d$OBTOTV24 + d$OPTOTV24 + d$ERTOT24 + d$IPDIS24 > 0)
d$erip <- as.numeric(d$ERTOT24 + d$IPDIS24 > 0)
d$unmet <- ifelse(d$DLAYCA42 %in% 1:2 & d$AFRDCA42 %in% 1:2, as.numeric(d$DLAYCA42 == 1 | d$AFRDCA42 == 1), NA)

des <- svydesign(ids = ~VARPSU, strata = ~VARSTR, weights = ~ESAQWT24F, nest = TRUE, data = d)
des <- subset(des, ESAQWT24F > 0)
cm <- subset(des, common)
tq <- qt(.975, degf(cm))
base <- "u*race + fb + age + I(age^2) + female + pov + ins"

report <- function(ds, y, extra, label) {
  m <- svyglm(as.formula(paste(y, "~", base, extra)), design = ds, family = quasibinomial())
  b <- coef(m); V <- vcov(m)
  lc <- function(nm) { L <- setNames(rep(0, length(b)), names(b)); L[nm] <- 1
    e <- sum(L * b); s <- sqrt(drop(t(L) %*% V %*% L)); sprintf("%.2f [%.2f, %.2f]", exp(e), exp(e - tq * s), exp(e + tq * s)) }
  nm <- c("u:raceHisp", "u:raceNHB")
  W <- drop(t(b[nm]) %*% solve(V[nm, nm]) %*% b[nm]); df2 <- degf(ds); Fval <- W / 2
  cat(sprintf("%-6s %-34s n=%5d | aOR NHW %s | Hisp %s | NHB %s | F(2,%d) = %.2f, p = %.4f | ROR Hisp %s | NHB %s\n",
      y, label, nrow(ds$variables), lc("u"), lc(c("u", "u:raceHisp")), lc(c("u", "u:raceNHB")), df2, Fval,
      pf(Fval, 2, df2, lower.tail = FALSE), lc("u:raceHisp"), lc("u:raceNHB")))
  invisible(m)
}

cat("== Table S1: sensitivity of the exposure-by-race interaction models to additional adjustment\n")
for (y in c("probpy", "debt", "sac")) {
  report(cm, y, "", "main model")
  report(cm, y, "+ lchg_own", "+ log10(1 + own total charges)")
  report(cm, y, "+ lchg_fam", "+ log10(1 + family total charges)")
  ce <- subset(cm, !is.na(edu))
  report(ce, y, "", "main model, valid-education sample")
  report(ce, y, "+ edu", "+ highest degree")
}
cat("\nUnweighted n by race, valid-education sample:\n"); print(table(d$race[d$common & !is.na(d$edu)]))

## Adjusted predicted percentages (standardized to the covariate distribution of the model sample)
pm <- function(m, ds, r, uu) {
  nd <- ds$variables; nd$race <- factor(r, levels = levels(d$race)); nd$u <- uu
  X <- model.matrix(delete.response(terms(m)), model.frame(delete.response(terms(m)), nd, xlev = m$xlevels))
  w <- weights(ds); w <- w / sum(w); p <- plogis(drop(X %*% coef(m)))
  list(est = sum(w * p), grad = colSums(w * p * (1 - p) * X))
}
cat("\n== Table S2: health care use and cost-related delayed or forgone care by hypothetical response\n")
cat("(adjusted predicted %, standardized to the model sample; gap vs NHW in percentage points [95% CI])\n")
for (y in c("visit", "erip", "unmet")) {
  ds <- if (y == "unmet") subset(cm, !is.na(unmet)) else cm
  m <- report(ds, y, "", "interaction model")
  V <- vcov(m)
  for (uu in 1:0) {
    w0 <- pm(m, ds, "NHW", uu); out <- sprintf("NHW %.1f", 100 * w0$est)
    for (r in c("Hisp", "NHB")) {
      g <- pm(m, ds, r, uu); e <- g$est - w0$est; gr <- g$grad - w0$grad; s <- sqrt(drop(t(gr) %*% V %*% gr))
      out <- c(out, sprintf("%s %.1f (gap %+.1f [%+.1f, %+.1f])", r, 100 * g$est, 100 * e, 100 * (e - tq * s), 100 * (e + tq * s)))
    }
    cat(sprintf("   %-6s u=%d  %s\n", y, uu, paste(out, collapse = " | ")))
  }
}
cat("\nUnweighted n with valid delayed/forgone-care items:", sum(d$common & !is.na(d$unmet)), "\n")
