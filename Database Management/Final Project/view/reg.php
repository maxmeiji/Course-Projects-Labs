
<html lang="zh-TW">
  <head>
    <title>註冊</title>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css"/>
    <link rel="stylesheet" href="css/style.css"/>
	<link rel="stylesheet" type="text/css" href="space.css" >
  </head>

  <body>

<div id="space"></div>

    <div class="content">
      <div class="container">
  
        <div class="row">
 
          <div class="col-xs-12 col-sm-4 col-sm-offset-4">
            <form class="register">
			 <div class="form-group">
                <label for="username">Lname</label>
                <input type="text" class="form-control" id="lname" name="lname" placeholder="請輸入名字" required>
              </div>
			  <div class="form-group">
                <label for="username">Fname</label>
                <input type="text" class="form-control" id="fname" name="fname" placeholder="請輸入姓氏" required>
              </div>
              
			  <div class="form-group">
                <label for="username">Username</label>
                <input type="text" class="form-control" id="username" name="username" placeholder="請輸入帳號" required>
              </div>
              <div class="form-group">
                <label for="password">Password</label>
                <input type="password" class="form-control" id="password" name="password" placeholder="請輸入密碼" required>
              </div>
              <div class="form-group">
                <label for="confirm_password">Confirm Password </label>
                <input type="password" class="form-control" id="confirm_password" name="password" placeholder="請再次輸入密碼" required>
              </div>
			  <div class="form-group">
                <label for="username">Card</label>
                <input type="text" class="form-control" id="card" name="card" placeholder="請輸入帳號" required>
              </div>
			  <div class="form-group">
                <label for="username">Phone-number</label>
                <input type="tel" class="form-control" id="phone" name="phone" placeholder="請輸入電話號碼"  required>
              </div>
			   <div class="form-group">
                <label for="username">Address</label>
                <input type="text" class="form-control" id="add" name="add" placeholder="請輸入地址"  required>
              </div>
			  
			  <!--<div class="form-group">
                <label for="name">身分</label>
                <select class="form-control" id="ident" placeholder="請勾選身分" required>
				<option>Buyer</option>
				<option>Seller</option>
				<option>Other</option>
				</select>
              </div>
              <div class="form-group">
                <label for="name">性別</label>
				 <select class="form-control" id="gender" placeholder="請勾選性別" required>
				<option>男</option>
				<option>女</option>
				<option>其它</option>
				</select>
              </div>-->
			  <div class="form-group">

			  
              <button type="submit" class="btn btn-default">
                註冊
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>

 
    <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
    <script>

     $(document).on("ready", function() {
			
			$("#username").on("keyup", function(){

      		var keyin_value = $(this).val();

      		if(keyin_value != '')
      		{
				
      			
	      		$.ajax({
			        type : "POST",	
			        url : "../check_username.php",  
			        data : {	
			          un: $(this).val()	
			        },
			        dataType : 'html' 
			      }).done(function(data) {
			        
			        console.log(data); 
			        if(data == "yes")
			        {
			        	
			        	$("#username").parent().removeClass("has-error").addClass("has-success"); 
			        	
			        	$("form.register button[type='submit']").removeClass('disabled');
			        }
			        else
			        {
			        	alert("帳號有重複，不可以註冊");
			        	$("#username").parent().removeClass("has-success").addClass("has-error");
			        	
                $("form.register button[type='submit']").addClass('disabled');
			        }
			        
			      }).fail(function(jqXHR, textStatus, errorThrown) {
			      	
			      	alert("有錯誤產生，請看 console log");
			        console.log(jqXHR.responseText);
			      });
      		}
      		else
      		{
      			
      			$("#username").parent().removeClass("has-success").removeClass("has-error");
      		}
      		
      	});
		$("#card").on("keyup", function(){
      		
      		var keyin_value = $(this).val();
      		
      		if(keyin_value != '')
      		{
      			
	      		$.ajax({
			        type : "POST",	
			        url : "../check_email.php",  
			        data : {	
			          cd: $(this).val()	
			        },
			        dataType : 'html' 
			      }).done(function(data) {
			       
			        console.log(data); 
			        if(data == "yes")
			        {
			        	
			        	$("#card").parent().removeClass("has-error").addClass("has-success"); 
			        	
			        	$("form.register button[type='submit']").removeClass('disabled');
			        }
			        else
			        {
			        	alert("卡號有重複，不可以註冊");
			        	$("#card").parent().removeClass("has-success").addClass("has-error");
			        	
                $("form.register button[type='submit']").addClass('disabled');
			        }
			        
			      }).fail(function(jqXHR, textStatus, errorThrown) {
			      	
			      	alert("有錯誤產生，請看 console log");
			        console.log(jqXHR.responseText);
			      });
      		}
      		else
      		{
      			
      			$("#card").parent().removeClass("has-success").removeClass("has-error");
      		}
      		
      	});
				
				$("form.register").on("submit", function(){
					console.log("1");
					
					if ($("#password").val() != $("#confirm_password").val()) {
						
						
						$("#password").parent().addClass("has-error"); 
						$("#confirm_password").parent().addClass("has-error"); 
	        	
	          alert("兩次密碼輸入不一樣，請確認");
	          
	        }
	        else
	        {
	        	
	      		$.ajax({
			        type : "POST",
			        url : "../add_user.php",
			        data : {
					  ln : $("#lname").val(),
					  fn : $("#fname").val(),
			          un : $("#username").val(), 
			          pw : $("#password").val(),
					  cd : $("#card").val(),					  
					  p : $("#phone").val(),
					 ad : $("#add").val(),

					  
			        },
			        dataType : 'html' 
			      }).done(function(data) {
			       
			        console.log(data);
			        if(data == "yes")
			        {
			          alert("註冊成功，將自動前往登入頁。");
			        	
			        	window.location.href="final.php";
			        }
			        else
			        {
			        	alert("註冊失敗，請與系統人員聯繫");
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