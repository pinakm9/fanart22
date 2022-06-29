(function($){

    
	$('#v11, #v12, #v13').hover(function toggleControls() {
		if (this.hasAttribute("controls")) {
			this.removeAttribute("controls")
		} else {
			this.setAttribute("controls", "controls")
		}
	});

})(jQuery);