# Reproduces the sample flow (Appendix Figure A1), Table 1 key rows, Table 2, Table 3, Appendix Table A1
# (labelled "Table 4" in the output), and ratio-of-odds-ratio contrasts for
# "Hypothetical Medical Bill Affordability and Experienced Financial Hardship by Race and Ethnicity" (MEPS 2024, HC-256).
# Usage: Rscript 01_survey_models.R path/to/input.dta
suppressMessages({library(haven); library(survey)})
args <- commandArgs(TRUE); stopifnot(length(args) == 1)
cat("R", R.version$major, ".", R.version$minor, " | survey", as.character(packageVersion("survey")),
    " | haven", as.character(packageVersion("haven")), "\n\n", sep = "")
v <- c("DUPERSID","VARSTR","VARPSU","ESAQWT24F","BORNUSA","RACETHX","EQPAYB53","PROBPY42","MEDDEBT42",
       "EQSNON53","AGE24X","SEX","POVCAT24","INSCOV24","RESP42","ESAQPROXY53")
d <- as.data.frame(zap_labels(read_dta(args[1], col_select = all_of(v))))

## ---- Sample flow (Appendix Figure A1) ----
s1 <- d$ESAQWT24F > 0
s2 <- s1 & d$BORNUSA %in% 1:2 & d$RACETHX %in% 1:4
s3 <- s2 & d$EQPAYB53 %in% 1:7
s4 <- s3 & d$PROBPY42 %in% 1:2
s5 <- s4 & d$MEDDEBT42 %in% 0:7
s6 <- s5 & d$EQSNON53 %in% 1:2
cat("== Sample flow\n"); print(c(records = nrow(d), esaq_weight = sum(s1), nativity_race = sum(s2),
  eqpayb_1to7 = sum(s3), probpy = sum(s4), meddebt = sum(s5), sacrifice = sum(s6)))
d$common <- s6
d$race <- factor(d$RACETHX, levels = c(2, 1, 3, 4), labels = c("NHW", "Hisp", "NHB", "NHA"))
cat("\nAnalytic n by race:\n"); print(table(d$race[d$common]))
cat("Unique person IDs in analytic sample:", length(unique(d$DUPERSID[d$common])), "\n")

d$u <- as.numeric(d$EQPAYB53 == 7)
d$probpy <- as.numeric(d$PROBPY42 == 1); d$debt <- as.numeric(d$MEDDEBT42 >= 1); d$sac <- as.numeric(d$EQSNON53 == 2)
d$age <- (d$AGE24X - 50) / 20; d$female <- as.numeric(d$SEX == 2)
d$fb <- factor(d$BORNUSA); d$pov <- relevel(factor(d$POVCAT24), ref = "5"); d$ins <- factor(d$INSCOV24)
d$selfr <- d$RESP42 == 1 & d$ESAQPROXY53 == 1
d$pay4 <- cut(d$EQPAYB53, c(0, 3, 5, 6, 7), labels = c("immediate", "card_borrow", "plan", "unable"))

# Full-sample design; analyses use subset() so variance reflects all strata/PSUs.
des <- svydesign(ids = ~VARPSU, strata = ~VARSTR, weights = ~ESAQWT24F, nest = TRUE,
                 data = d[d$ESAQWT24F > 0 | TRUE, ])
des <- subset(des, ESAQWT24F > 0)
cm <- subset(des, common)
cat("Design degrees of freedom (analytic subset):", degf(cm), "\n")

## ---- Table 1 (selected rows, weighted %) ----
cat("\n== Table 1: weighted % by race\n")
t1 <- svyby(~I(u) + probpy + debt + sac + female, ~race, cm, svymean)
print(round(100 * t1[, c("I(u)", "probpy", "debt", "sac", "female")], 1))
print(round(100 * svymean(~pay4, cm), 1))

## ---- Table 3: prevalence of outcomes by hypothetical response, logit CIs ----
cat("\n== Table 3: weighted % [95% CI] of each outcome among unable (u=1) / able (u=0)\n")
for (y in c("probpy", "debt", "sac")) {
  for (uu in 1:0) {
    r <- svyby(as.formula(paste0("~", y)), ~race, subset(cm, u == uu & race != "NHA"),
               svyciprop, vartype = "ci", method = "logit")
    n <- table(d$race[d$common & d$u == uu])[c("NHW", "Hisp", "NHB")]
    cat(sprintf("%-6s u=%d  %s\n", y, uu, paste(sprintf("%s %.1f [%.1f, %.1f] n=%d",
        rownames(r), 100 * r[[y]], 100 * r$ci_l, 100 * r$ci_u, n[rownames(r)]), collapse = " | ")))
  }
}

## ---- Tables 2 and 4: interaction models ----
fit_report <- function(ds, label) {
  cat(sprintf("\n== %s (unweighted n = %d; df = %d)\n", label, nrow(ds$variables), degf(ds)))
  for (y in c("probpy", "debt", "sac")) {
    m <- svyglm(as.formula(paste(y, "~ u*race + fb + age + I(age^2) + female + pov + ins")),
                design = ds, family = quasibinomial())
    b <- coef(m); V <- vcov(m); tq <- qt(.975, degf(ds))
    lc <- function(nm) { L <- setNames(rep(0, length(b)), names(b)); L[nm] <- 1
      e <- sum(L * b); s <- sqrt(drop(t(L) %*% V %*% L)); sprintf("%.2f [%.2f, %.2f]", exp(e), exp(e - tq * s), exp(e + tq * s)) }
    nm <- c("u:raceHisp", "u:raceNHB")
    W <- drop(t(b[nm]) %*% solve(V[nm, nm]) %*% b[nm]); df2 <- degf(ds)
    Fval <- W / 2; p <- pf(Fval, 2, df2, lower.tail = FALSE)
    cat(sprintf("%-6s aOR NHW %s | Hisp %s | NHB %s | Wald F(2,%d) = %.2f, p = %.4f | ROR Hisp %s | ROR NHB %s\n",
        y, lc("u"), lc(c("u", "u:raceHisp")), lc(c("u", "u:raceNHB")), df2, Fval, p, lc("u:raceHisp"), lc("u:raceNHB")))
  }
}
fit_report(cm, "Table 2: full analytic sample")
sr <- subset(cm, selfr)
cat("\nSelf-respondent n by race:\n"); print(table(d$race[d$common & d$selfr]))
fit_report(sr, "Table 4: household respondent & ESAQ self-respondent")
