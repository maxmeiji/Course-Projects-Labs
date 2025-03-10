<?php

	require_once '../db.php';

?>
<head>
    <title>更改密碼</title>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css"/>
	<link rel="stylesheet" type="text/css" href="space.css" >

  </head>
 <body>

 <div id="space"></div>
<div class="content">
      <div class="container">
        
        <div class="row">
          
          <div class="col-xs-12 col-sm-4 col-sm-offset-4">
            <form class="check">
              <div class="form-group">
                <label for="old">舊密碼</label>
                <input type="password" class="form-control" id="old" name="old" placeholder="請輸入舊密碼" required>
              </div>
			  <div class="form-group">
                <label for="new">輸入新密碼</label>
                <input type="password" class="form-control" id="new" name="new" placeholder="請輸入新密碼" required>
              </div>
              <div class="form-group">
                <label for="password">確認新密碼</label>
                <input type="password" class="form-control" id="newcheck" name="newcheck" placeholder="請確認新密碼" required>
              </div>


			  
              <button type="submit" class="btn btn-default">
                更改密碼
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
	</div>
<script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
    <script>
		$(document).on("ready", function() {
			
				
				$("form.check").on("submit", function(){
					console.log("1");
					
					if ($("#new").val() != $("#newcheck").val()) {
						
						$("#new").parent().addClass("has-error"); 
						$("#newcheck").parent().addClass("has-error"); 
	        	
	          alert("兩次密碼輸入不一樣，請確認");
	          
	        }
	        else
	        {
	        	
	      		$.ajax({
			        type : "POST",
			        url : "../check_change.php",
			        data : {
					  op : $("#old").val(),
			          np : $("#new").val(), 

			        },
			        dataType : 'html' 
			      }).done(function(data) {
			        
			        console.log(data);
			        if(data == "yes")
			        {
			          alert("更新成功，將自動前往登入頁。");
			        	
			        	window.location.href="final.php";
			        }
			        else
			        {
			        	alert("舊密碼輸入錯誤");
			        }
			        
			      }).fail(function(jqXHR, textStatus, errorThrown) {
			      	
			      	alert("有錯誤產生，請看 console log");
			        console.log(jqXHR.responseText);
			      });
	        }
	        
	        
	        
          return false;
				});
      });
    </script>
	</body>