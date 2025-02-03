####################################################################################
#####################################    CLASE 1    ################################    
####################################################################################

# SUBIR LA BASE DE DATOS
BASE1
CASO 1

# VER NOMBRES DE LAS VARIABLES DE LA BASE 
ls(BASE1)
names(BASE1)
summary(BASE1)
mean(BASE1$PBI)
sd(BASE1$PBI)
hist(BASE1$PBI)

# CREAR SUB BASE DE DATOS ay un lado para columnas y hay otro para las columnas
data1 <- BASE1[,c("PBI","InvPri")]
data2 <- BASE1[,c("PBI","InvPu")]
data3 <- BASE1[,c("PBI","Exp")]
data6 <- BASE1[,c("InvPri","InvPu")]
data4<-BASE1[,c("PBI","InvPri","ConsPu","Exp")]
data5<-BASE1[,c("PBI","InvPri","Imp","Exp")]
# GRAFICAS  tener en cuenta que la instalacion se hace solo una vez
install.packages("ggplot2")
library("ggplot2")

plot(data1)
plot(data3)

# agregar x11 y hacerlo correr todo junto mas este plot(data1)

grafica1 <- ggplot(BASE1,aes(PBI,InvPri)) + geom_point() + geom_smooth(method = "lm", color = "red")
grafica2 <- ggplot(BASE1,aes(PBI,Exp)) + geom_point()  + geom_smooth(method = "lm", color = "blue")
grafica3 <- ggplot(BASE1,aes(PBI,InvPu)) + geom_point() + geom_smooth(method = "lm", color = "green")
grafica4 <- ggplot(data6,aes(InvPri,InvPu)) + geom_point() + geom_smooth(method = "lm", color = "pink")
# MODELO - REGRESION - ECUACION
names(data1)

# MODELO UNIVARIADO
regresion1 <- lm(PBI~InvPri+0, data = data1)
summary(regresion1)

PBI^ = 1.97454 + 0.26889*InvPri + u

# primer punto para el analizis de la regresion que el coeficiente tenga sentido economico 
# segundo punto el p valor tiene que ser menor a 0.05 ,o si aparece los asteristicos,paso la prueba de hipotesis
# tercero,el r al cuadrado indica que es un buen modelo mientras mas alto sea mejor

regresion3 <- lm(PBI~Exp, data = data3)
summary(regresion3)

PBI^ = 2.37660 + 0.61887*Exp + u


# MODELO MULTIVARIADO
regresion2 <- lm(PBI~InvPri+ConsPri, data = BASE1)
summary(regresion2)

PBI^ = -0.72598 + 0.08254*InvPri + 0.92427*ConsPri + u

regresion4 <- lm(PBI~Exp+Imp+InvPri, data = BASE1)
summary(regresion4)

PBI^ = -1.51817 + 0.08254*InvPri + 0.92427*ConsPri + u

# Coef. Correlacion puede ser negativo
cor(data1)
R = 0.8917163   ... 89.17%
se observan un coeficiente de correlacion de 89.17%, eso quiere decir que las variable inversion privada, tienen un grado de relacion alto y positiva .Eso quiere decir que cuando la cuando la inversion privada sube el pbi sube  
pairs(data1)

# habia un problema con la datam por lo que no recocia
BASE1num <- BASE1[ , sapply(BASE1, is.numeric)]
cor(BASE1num)
cor(BASE1)

#por lo que se evidencia que la variabl que tiene un alto grado de relacion con el PBI es la inversion y consumo privado

# grafica de correlacion con % no puede llegar a ser negativo
install.packages("psych")
library(psych)
pairs.panels(data1)
x11
pairs.panels(BASE1)

# Coef. Determinacion
R2 <- cor(data1)^2
R^2 = 0.7952  .... 79.52%
se observa que la invprivada influye, impacta o determina al PBI en 79.59%. Y el resto 20.48% reprecenta la influencia de otras variables que no se incluyen en el modelo 

# INTERVALOS DE CONFIANZA  nos muestra donde esta el verdadero beta poblacional 
confint(regresion1)

confint(regresion1, level = 0.90)
confint(regresion1, level = 0.95)
confint(regresion1, level = 0.99)

# ANOVA varianzas , que las varianzas de las variables sean heterogenea osea diferentes,poerque 
anova1 <- aov(regresion1)
summary(anova1)

# PRONOSTICO que pasaria si la inversion CRECER en 6 que pasaria cuanto seria el PBI
pronostico_PBI <- data.frame(InvPri = 10)
predict(regresion1, pronostico_PBI)

# A MANO
PBI = 1.97454 + 0.26889*6
