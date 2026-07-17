const imageInput=document.getElementById("image");
const preview=document.getElementById("preview");
const loading=document.getElementById("loading");

imageInput.onchange=function(){

    const file=this.files[0];

    if(file){

        preview.src=URL.createObjectURL(file);

        preview.style.display="block";

    }

}

document.querySelector("form").onsubmit=function(){

    loading.style.display="block";

}